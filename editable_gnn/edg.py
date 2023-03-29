import torch
import torch.nn as nn
from torch_geometric.data.data import Data
import numpy as np
import torch.nn.functional as F
from qpsolvers import solve_qp

class EDG(nn.Module):
    def __init__(self, model_config, loss_op, args):
        self.model_config = model_config
        self.gamma = args.gamma
        self.train_grads = {}
        self.target_grads = {}
        self.loss_op = loss_op
    

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project_grad(self, grad_target, grad_train):
        """
            Gradient refinement given a target editable sample loss
            gradient "gradient", and a training task loss gradients "grad_train".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  grad_train, p-vector
            output: x, p-vector
        """
        
        grad_train_np = grad_train.cpu().contiguous().double().numpy()
        grad_target_np = grad_target.cpu().contiguous().double().numpy()
        
        #print(grad_train_np.shape, grad_target_np.shape)
        v = np.dot(grad_train_np.transpose(), grad_target_np) / np.dot(grad_train_np.transpose(), grad_train_np)
        v = max(v, 0)
    
        x = np.dot(v, grad_train_np) + (1 + self.gamma)**(-1) * grad_target_np
        new_grad = torch.Tensor(x).view(-1)
        
        new_grad = new_grad.cuda()
        return new_grad
    
    def grab_input(self, data: Data, indices=None):
        # print(f'data={data}')
        if self.model_config['arch_name'] == 'MLP':
            return {"x": data.x}
        else:
            return {"x": data.x, 'adj_t': data.adj_t}

    def update_model(self, model, train_data, whole_data, idx, label, optimizer, max_num_step):
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}  # For convenience

        for step in range(1, max_num_step + 1):
            # compute gradient on training data     
            optimizer.zero_grad()
            input = self.grab_input(train_data)
            out = model(**input)
            loss = self.loss_op(out[train_data.train_mask], train_data.y[train_data.train_mask])
            loss.backward()
            self.grad_train = self.grad_to_vector()
            
            # compute gradient on target node 
            optimizer.zero_grad()
            input = self.grab_input(whole_data)
            # input['x'] = input['x']
            out = model(**input)[idx]
            loss = self.loss_op(out, label)
            loss.backward()
            self.grad_target = self.grad_to_vector()

            # gradient refinement   
            new_grad = self.project_grad(self.grad_target, self.grad_train)
            # copy gradients back
            self.vector_to_grad(new_grad)

            optimizer.step()

            y_pred = out.argmax(dim=-1)
            # sequential or independent setting
            if label.shape[0] == 1:
                if y_pred == label:
                    success = True
                    break
                else:
                    success = False
            # batch setting
            else:
                success = int(y_pred.eq(label).sum()) / label.size(0)
                if success == 1.:
                    break
        return model, success, loss, step

class EDG_Plus(EDG):
    def __init__(self, model_config, loss_op, args):
        super(EDG_Plus, self).__init__(
            model_config=model_config, 
            loss_op=loss_op, 
            args=args)

        self.model_config = model_config
        self.gamma = args.gamma
        self.train_split = args.train_split

        self.loss_op = loss_op
    
    def project_grad(self, grad_target, grad_train):
        """
            Gradient refinement given a target editable sample loss
            gradient "gradient", and a training task loss gradients "grad_train".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  grad_train, K * p-matrix
            output: x, p-vector

            Quadratic programming: https://qpsolvers.github.io/qpsolvers/quadratic-programming.html
        """
        
        grad_train_np = grad_train.cpu().contiguous().double().numpy()
        grad_target_np = grad_target.cpu().contiguous().double().numpy()

        K = grad_train_np.shape[0]
        
        #print(grad_train_np.shape, grad_target_np.shape)
        P = np.dot(grad_train_np, grad_train_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(grad_train_np, grad_target_np)

        # G = None
        # h = None  
        # A = None
        # b = None 

        lb = np.zeros(K)
        # ub = None

        v = solve_qp(P, q, lb=lb, solver="osqp")
        x = np.dot(grad_train_np.transpose(), v) + grad_target_np 
        x = (1 + self.gamma)**(-1) * x
        new_grad = torch.Tensor(x).view(-1)
        
        new_grad = new_grad.cuda()
        return new_grad
    
    def update_model(self, model, train_data, whole_data, idx, label, optimizer, max_num_step):
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}  # For convenience

        # print(f'whole_data={whole_data}')
        # print(f'train_data={train_data}')

        num_nodes = train_data.train_mask.shape[0]
        # print(f'num_nodes={num_nodes}')
        # print(f'train_data.train_mask={train_data.train_mask}')
        true_index = torch.nonzero(train_data.train_mask)
        # print(f'true_index={true_index}')
        true_indexes = np.array_split(np.array(range(true_index.shape[0])), self.train_split)
        # print(f'true_indexes={len(true_indexes[0])}')
        # print(f'true_indexes={len(true_indexes[1])}')
        train_sub_masks = torch.zeros((self.train_split, num_nodes), dtype=torch.bool)

        # print(f'train_sub_masks={train_sub_masks}')
        for i in range(self.train_split):
            train_sub_masks[i, true_indexes[i]] = True



        # print(f'train_mask={train_data.train_mask}')
        # print(f'train_sub_masks={train_sub_masks}')
        for step in range(1, max_num_step + 1):
            # compute gradient on training data 
            grad_train = {} 
            for k in range(self.train_split):   
                optimizer.zero_grad()
                input = self.grab_input(train_data)
                out = model(**input)
                # print(f'out={out[train_sub_masks[k]].shape}')
                loss = self.loss_op(out[train_sub_masks[k]], train_data.y[train_sub_masks[k]])
                loss.backward()
                grad_train[k] = self.grad_to_vector()
            grad_train = torch.stack(list(grad_train.values()))

            # compute gradient on target node 
            optimizer.zero_grad()
            input = self.grab_input(whole_data)
            # input['x'] = input['x']
            out = model(**input)[idx]
            loss = self.loss_op(out, label)
            loss.backward()
            grad_target = self.grad_to_vector()

            # gradient refinement   
            new_grad = self.project_grad(grad_target, grad_train)
            # copy gradients back
            self.vector_to_grad(new_grad)

            optimizer.step()

            y_pred = out.argmax(dim=-1)
            # sequential or independent setting
            if label.shape[0] == 1:
                if y_pred == label:
                    success = True
                    break
                else:
                    success = False
            # batch setting
            else:
                success = int(y_pred.eq(label).sum()) / label.size(0)
                if success == 1.:
                    break
        return model, success, loss, step