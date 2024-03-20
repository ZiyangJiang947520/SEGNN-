import os
import time
import pdb
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import re
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.data.data import Data
from editable_gnn.models.base import BaseModel
from editable_gnn.logger import Logger
from editable_gnn.utils import set_seeds_all, kl_logit, ada_kl_logit
from editable_gnn.edg import EDG, EDG_Plus
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class BaseTrainer(object):
    def __init__(self,
                 args,
                 model: BaseModel,
                 train_data: Data,
                 whole_data: Data,
                 model_config: Dict,
                 output_dir: str,
                 dataset_name: str,
                 is_multi_label_task: bool,
                 amp_mode: bool = False,
                 load_pretrained_backbone: bool = False) -> None:
        self.model = model
        self.train_data = train_data
        self.whole_data = whole_data
        self.model_config = model_config
        self.model_name = model_config['arch_name']
        if amp_mode is True:
            raise NotImplementedError

        self.runs = args.runs
        self.logger = Logger(args.runs)


        self.optimizer = None
        self.save_path = os.path.join(output_dir, dataset_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.loss_op = F.binary_cross_entropy_with_logits if is_multi_label_task else F.cross_entropy


        self.seed = args.seed

        self.gamma = args.gamma if hasattr(args, 'gamma') else 1.0
        self.args = args
        self.hyper_Diff = args.hyper_Diff if hasattr(args, 'hyper_Diff') else 0.0
        self.load_pretrained_backbone = load_pretrained_backbone
        self.num_mixup_training_samples = args.num_mixup_training_samples
        self.between_edit_ftn = args.finetune_between_edit
        self.stop_edit_only = args.stop_edit_only
        self.iters_before_stop = args.iters_before_stop
        self.full_edit = args.full_edit
        self.mixup_k_nearest_neighbors = args.mixup_k_nearest_neighbors
        self.incremental_batching = args.incremental_batching
        self.sliding_batching = args.sliding_batching
        self.stop_full_edit = args.stop_full_edit
        self.half_half = args.half_half
        self.pure_egnn = args.pure_egnn
        self.half_half_ratio_mixup = args.half_half_ratio_mixup
        self.wrong_ratio_mixup = args.wrong_ratio_mixup


    def train_loop(self,
                   model: BaseModel,
                   optimizer: torch.optim.Optimizer,
                   train_data: Data,
                   loss_op):
        model.train()
        optimizer.zero_grad()
        input = self.grab_input(train_data)
        out = model(**input)
        loss = loss_op(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()


    def train(self):
        for run in range(self.runs):
            set_seeds_all(self.seed + run)
            self.single_run(run)
        self.logger.print_statistics()


    def save_model(self, checkpoint_prefix: str, epoch: int):
        best_model_checkpoint = os.path.join(self.save_path, f'{checkpoint_prefix}_{epoch}.pt')
        torch.save(self.model.state_dict(), best_model_checkpoint)
        checkpoints_sorted = self.sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, self.save_path)
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            os.remove(f'./{checkpoint}')


    def single_run(self, run: int):
        if not self.load_pretrained_backbone:
            self.model.reset_parameters()
        optimizer = self.get_optimizer(self.model_config, self.model)
        best_val = -1.
        checkpoint_prefix = f'{self.model_name}_run{run}'
        for epoch in range(1, self.model_config['epochs'] + 1):
            if not self.load_pretrained_backbone:
                train_loss = self.train_loop(self.model, optimizer, self.train_data, self.loss_op)
            result = self.test(self.model, self.whole_data)
            self.logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            # save the model with the best valid acc
            if valid_acc > best_val:
                self.save_model(checkpoint_prefix, epoch)
                best_val = valid_acc

            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train f1: {100 * train_acc:.2f}%, '
                    f'Valid f1: {100 * valid_acc:.2f}% '
                    f'Test f1: {100 * test_acc:.2f}%')
        self.logger.print_statistics(run)


    @staticmethod
    def compute_micro_f1(logits, y, mask=None) -> float:
        if mask is not None:
            logits, y = logits[mask], y[mask]
        if y.dim() == 1:
            try:
                return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
            except ZeroDivisionError:
                return 0.

        else:
            y_pred = logits > 0
            y_true = y > 0.5

            tp = int((y_true & y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())

            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                return 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                return 0.


    @torch.no_grad()
    def test(self, model: BaseModel, data: Data, specific_class: int = None):
        model.eval()
        out = self.prediction(model, data)
        y_true = data.y
        train_mask = data.train_mask
        valid_mask = data.val_mask
        test_mask = data.test_mask
        if specific_class is not None:
            mask = data.y == specific_class
            out = out[mask]
            y_true = y_true[mask]
            train_mask = train_mask[mask]
            valid_mask = valid_mask[mask]
            test_mask = test_mask[mask]
        train_acc = self.compute_micro_f1(out, y_true, train_mask)
        valid_acc = self.compute_micro_f1(out, y_true, valid_mask)
        test_acc = self.compute_micro_f1(out, y_true, test_mask)
        return train_acc, valid_acc, test_acc


    @torch.no_grad()
    def prediction(self, model: BaseModel, data: Data):
        model.eval()
        input = self.grab_input(data)
        return model(**input)


    @staticmethod
    def sorted_checkpoints(
        checkpoint_prefix, best_model_checkpoint, output_dir=None, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
            checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted


    @staticmethod
    def get_optimizer(model_config, model):
        if model_config['optim'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
        elif model_config['optim'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
        else:
            raise NotImplementedError
        # if model_config['optim'] == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
        # elif model_config['optim'] == 'rmsprop':
        #     optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
        # else:
        #     raise NotImplementedError
        return optimizer


    def select_node(self, whole_data: Data,
                    num_classes: int,
                    num_samples: int,
                    criterion: str,
                    from_valid_set: bool = True):
        self.model.eval()
        bef_edit_logits = self.prediction(self.model, whole_data)
        bef_edit_pred = bef_edit_logits.argmax(dim=-1)
        val_y_true = whole_data.y[whole_data.val_mask]
        val_y_pred = bef_edit_pred[whole_data.val_mask]
        if from_valid_set:
            nodes_set = whole_data.val_mask.nonzero().squeeze()
        else:
            # select from the train set
            nodes_set = whole_data.train_mask.nonzero().squeeze()
        assert criterion in ['wrong2correct', 'random']
        if criterion == 'wrong2correct':
            wrong_pred_set = val_y_pred.ne(val_y_true).nonzero()
            val_node_idx_2flip = wrong_pred_set[torch.randperm(len(wrong_pred_set))[:num_samples]]
            node_idx_2flip = nodes_set[val_node_idx_2flip]
            flipped_label = whole_data.y[node_idx_2flip]
        elif criterion == 'random':
            node_idx_2flip = nodes_set[torch.randint(high=len(nodes_set), size=(num_samples, 1))]
            flipped_label = torch.randint(high=num_classes, size=(num_samples, 1))
        else:
            raise NotImplementedError
        return node_idx_2flip, flipped_label

    def select_mixup_training_nodes(self,
                                    whole_data: Data,
                                    criterion: str,
                                    num_samples:int = 0,
                                    center_node_idx=None):
        self.model.eval()
        bef_edit_logits = self.prediction(self.model, whole_data)
        bef_edit_pred = bef_edit_logits.argmax(dim=-1)
        train_y_true = whole_data.y[whole_data.train_mask]
        train_y_pred = bef_edit_pred[whole_data.train_mask]
        nodes_set = whole_data.train_mask.nonzero().squeeze()
        right_pred_set = mixup_training_samples_idx = None

        assert criterion in ['wrong2correct', 'random']
        if criterion == 'wrong2correct':
            right_pred_set = train_y_pred.eq(train_y_true).nonzero()
            dvc = right_pred_set.device
            if center_node_idx != None:
                neighbors = torch.Tensor([])
                num_hop = 0
                while len(neighbors) < num_samples and num_hop < 4:
                    num_hop += 1
                    neighbors, _, _, _ = k_hop_subgraph(center_node_idx, num_hops=num_hop, edge_index=self.whole_data.edge_index)
                right_pred_set = right_pred_set.squeeze().cpu().numpy().tolist()
                #select wrong samples for mixup
                if self.wrong_ratio_mixup > 0:
                    right_pred_set = torch.zeros(len(whole_data.y)).bool().to(dvc)
                    for i in neighbors:
                        right_pred_set[i] = True
                    right_pred_set = right_pred_set[whole_data.train_mask].nonzero().type(torch.LongTensor).to(dvc)
                else:
                #pdb.set_trace()
                    right_pred_set = torch.Tensor([int(i) for i in right_pred_set if i in neighbors]).unsqueeze(dim=1).type(torch.LongTensor).to(dvc)
                #right_pred_set = neighbors.unsqueeze(dim=1).type(torch.LongTensor).to(dvc)

            half_half = self.half_half
            if half_half:
                train_pred_set = train_y_pred.eq(train_y_true).nonzero().to(dvc)
                train_mixup_training_samples_idx = torch.cat((
                                                            right_pred_set[torch.randperm(len(right_pred_set))[:int(num_samples * self.half_half_ratio_mixup)]].type(torch.LongTensor).to(dvc),
                                                            train_pred_set[torch.randperm(len(train_pred_set))[int(num_samples * self.half_half_ratio_mixup):num_samples]]), dim=0)
            else:
                train_mixup_training_samples_idx = right_pred_set[torch.randperm(len(right_pred_set))[:num_samples]].type(torch.LongTensor)

            #pdb.set_trace()
            mixup_training_samples_idx = nodes_set[train_mixup_training_samples_idx]
            mixup_label = whole_data.y[mixup_training_samples_idx]

        return mixup_training_samples_idx, mixup_label

    def single_edit(self, model, idx, label, optimizer, max_num_step, time_to_full_edit = False, num_edit_targets=1, pure_egnn_edit=False):
        model.train()
        s = time.time()
        torch.cuda.synchronize()
        for step in range(1, max_num_step + 1):
            optimizer.zero_grad()
            input = self.grab_input(self.whole_data)
            input['x'] = input['x'][idx]
            out = model(**input)
            loss = self.loss_op(out, label)
            loss.backward()
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
                if self.stop_edit_only:
                    success = int(y_pred[:num_edit_targets].eq(label[:num_edit_targets])[:num_edit_targets].sum()) / num_edit_targets
                else:
                    success = int(y_pred.eq(label).sum()) / label.size(0)
                if success == 1.:
                    break
        torch.cuda.synchronize()
        e = time.time()
        print(f'edit time: {e - s}')
        return model, success, loss, step

    def single_Diff_edit(self, model, idx, label, optimizer, max_num_step):
        input = self.grab_input(self.whole_data)
        with torch.no_grad():
            out_ori_val = model(**input)[self.whole_data.val_mask]
        for step in range(1, max_num_step + 1):
            optimizer.zero_grad()
            input = self.grab_input(self.whole_data)
            out = model(**input)
            # pdb.set_trace()
            kl_loss = kl_logit(out[self.whole_data.val_mask], out_ori_val)
            # print(f'args={self.args}')
            loss = self.loss_op(out[idx], label) + self.hyper_Diff * kl_loss
            # pdb.set_trace()
            loss.backward()
            optimizer.step()
            y_pred = out[idx].argmax(dim=-1)
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

    def single_Diff_Ada_edit(self, model, idx, label, optimizer, max_num_step):
        input = self.grab_input(self.whole_data)
        with torch.no_grad():
            out_ori_val = model(**input)[self.whole_data.val_mask]

        for step in range(1, max_num_step + 1):
            optimizer.zero_grad()
            input = self.grab_input(self.whole_data)
            out = model(**input)
            loss = self.loss_op(out[idx], label) + self.hyper_Diff * ada_kl_logit(out[self.whole_data.val_mask], out_ori_val, self.gamma)
            loss.backward()
            optimizer.step()
            y_pred = out[idx].argmax(dim=-1)
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

    def EDG_edit(self, model, idx, label, optimizer, max_num_step):
        edg = EDG(self.model_config, self.loss_op, self.args)
        model, success, loss, step = edg.update_model(model, self.train_data, self.whole_data, idx, label, optimizer, max_num_step)

        return model, success, loss, step

    def EDG_Plus_edit(self, model, idx, label, optimizer, max_num_step):
        edg_plus = EDG_Plus(self.model_config, self.loss_op, self.args)
        model, success, loss, step = edg_plus.update_model(model, self.train_data, self.whole_data, idx, label, optimizer, max_num_step)

        return model, success, loss, step

    def bef_edit_check(self,  model, idx, label, curr_edit_target):
        model.eval()
        torch.cuda.synchronize()
        input = self.grab_input(self.whole_data)
        if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GCN2_MLP', 'GAT_MLP', 'JKNET_MLP', 'GIN_MLP']:
            out = model.fast_forward(input['x'][idx], idx)
            y_pred = out.argmax(dim=-1)
        else:
            out = model(**input)
            y_pred = out.argmax(dim=-1)[idx]

        # sequential or independent setting
        if label.shape[0] == 1:
            if y_pred == label:
                success = True
            else:
                success = False
        # batch setting
        else:
            success = 1.0 if y_pred.eq(label)[curr_edit_target] else 0.0
        torch.cuda.synchronize()
        return success

    def edit_select(self, model, idx, f_label, optimizer, max_num_step, manner='GD',
                    mixup_training_samples_idx = torch.Tensor([]), time_to_full_edit = False,
                    num_edit_targets=1, curr_edit_target=0, pure_egnn_edit=False):
        bef_edit_success = self.bef_edit_check(model, idx, f_label,curr_edit_target=curr_edit_target)
        if bef_edit_success == 1.:
            return model, bef_edit_success, 0, 0

        assert manner in ['GD', 'GD_Diff', 'Ada_GD_Diff', 'EDG', 'EDG_Plus']

        random_sampling = False
        if self.between_edit_ftn and model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GCN2_MLP', 'GAT_MLP', 'JKNET_MLP', 'GIN_MLP'] and (random_sampling or mixup_training_samples_idx.size(0) > 0):
            #pdb.set_trace()
            self.between_edit_finetune_mlp(batch_size=50, iters=100, idx=mixup_training_samples_idx.squeeze(dim=1), random_sampling=random_sampling)

        if manner == 'GD':
            return self.single_edit(model, idx, f_label, optimizer, max_num_step,
                                    time_to_full_edit = time_to_full_edit, num_edit_targets=num_edit_targets,
                                    pure_egnn_edit=pure_egnn_edit)
        elif manner == 'GD_Diff':
            return self.single_Diff_edit(model, idx, f_label, optimizer, max_num_step)
        elif manner == 'Ada_GD_Diff':
            return self.single_Diff_Ada_edit(model, idx, f_label, optimizer, max_num_step)
        elif manner == 'EDG':
            return self.EDG_edit(model, idx, f_label, optimizer, max_num_step)
        else:
            return self.EDG_Plus_edit(model, idx, f_label, optimizer, max_num_step)



    def sequential_edit(self, node_idx_2flip, flipped_label, whole_data, max_num_step, manner='GD', specific_class=None):
        self.model.train()
        model = deepcopy(self.model)
        optimizer = self.get_optimizer(self.model_config, model)
        results_temporary = []
        i = 0
        for idx, f_label in tqdm(zip(node_idx_2flip, flipped_label)):
            i = i + 1
            # edited_model, success, loss, steps = self.single_edit(model, idx, f_label, optimizer, max_num_step)
            edited_model, success, loss, steps = self.edit_select(model, idx, f_label, optimizer, max_num_step, manner)
            success = self.success_rate(model, node_idx_2flip[:i].squeeze(dim=1), flipped_label[:i].squeeze(dim=1))
            if specific_class is None:
                res = [*self.test(edited_model, whole_data), success, steps]
            else:
                res = [*self.test(edited_model, whole_data),
                       *self.test(edited_model, whole_data, specific_class=specific_class),
                       success,
                       steps]
            # for n_hop in [1, 2]:
            #     res.append(self.get_khop_neighbors_acc(model, n_hop, idx))
            results_temporary.append(res)
        return results_temporary


    def independent_edit(self, node_idx_2flip, flipped_label, whole_data, max_num_step, num_htop=0, manner='GD', specific_class=None):
        self.model.train()
        results_temporary = []
        for idx, f_label in tqdm(zip(node_idx_2flip, flipped_label)):
            model = deepcopy(self.model)
            optimizer = self.get_optimizer(self.model_config, model)
            # edited_model, success, loss, steps = self.single_edit(model, idx, f_label, optimizer, max_num_step)
            edited_model, success, loss, steps = self.edit_select(model, idx, f_label, optimizer, max_num_step, manner)
            if specific_class is None:
                res = [*self.test(edited_model, whole_data), success, steps]
            else:
                res = [*self.test(edited_model, whole_data),
                       *self.test(edited_model, whole_data, specific_class=specific_class),
                       success,
                       steps]
            # import ipdb; ipdb.set_trace()
            # torch.save(edited_model.state_dict(), 'cora_gcn_mlp.pt')
            hop_res = []
            for n_hop in range(1, num_htop+1):
                hop_res.append(self.get_khop_neighbors_acc(model, n_hop, idx))
            res.append(hop_res)
            results_temporary.append(res)
        return results_temporary


    def batch_edit(self, node_idx_2flip, flipped_label, whole_data, max_num_step, num_htop=0, manner='GD', mixup_training_samples_idx = None, mixup_label = None):
        self.model.train()
        model = deepcopy(self.model)
        optimizer = self.get_optimizer(self.model_config, model)
        results_temporary = []
        #pdb.set_trace()
        for idx in tqdm(range(len(node_idx_2flip))):
            set_seeds_all(idx)
            if self.mixup_k_nearest_neighbors:
                mixup_training_samples_idx, mixup_label = self.select_mixup_training_nodes(self.whole_data,
                                                                                        'wrong2correct',
                                                                                        num_samples = self.num_mixup_training_samples,
                                                                                        center_node_idx=node_idx_2flip[idx])
            else:
                mixup_training_samples_idx, mixup_label = self.select_mixup_training_nodes(self.whole_data,
                                                                                        'wrong2correct',
                                                                                        num_samples = self.num_mixup_training_samples)
            if mixup_training_samples_idx is not None:
                nodes = torch.Tensor([])
                labels = torch.Tensor([])
                if self.incremental_batching:
                    nodes = torch.cat((node_idx_2flip[:idx+1].squeeze(dim=1), mixup_training_samples_idx.squeeze(dim=1)), dim=0)
                    labels = torch.cat((flipped_label[:idx+1].squeeze(dim=1), mixup_label.squeeze(dim=1)), dim=0)
                    num_edit_targets = idx + 1
                elif self.sliding_batching > 0:
                    nodes = torch.cat((node_idx_2flip[idx:min(idx + self.sliding_batching, len(node_idx_2flip))].squeeze(dim=1), mixup_training_samples_idx.squeeze(dim=1)), dim=0)
                    labels = torch.cat((flipped_label[idx:min(idx + self.sliding_batching, len(node_idx_2flip))].squeeze(dim=1), mixup_label.squeeze(dim=1)), dim=0)
                    num_edit_targets = min(self.sliding_batching, len(node_idx_2flip) - idx)
                else:
                    nodes = torch.cat((node_idx_2flip[idx], mixup_training_samples_idx.squeeze(dim=1)), dim=0)
                    labels = torch.cat((flipped_label[idx], mixup_label.squeeze(dim=1)), dim=0)
                    num_edit_targets = 1
                edited_model, success, loss, steps = self.edit_select(model,
                                                                    nodes,
                                                                    labels,
                                                                    optimizer,
                                                                    max_num_step,
                                                                    manner = manner,
                                                                    mixup_training_samples_idx = torch.Tensor([]),
                                                                    time_to_full_edit = (idx > 0 and self.full_edit > 0 and (idx + 1) % self.full_edit == 0),
                                                                    num_edit_targets=num_edit_targets,
                                                                    curr_edit_target=idx,
                                                                    pure_egnn_edit=(self.pure_egnn > idx))
            else:
                edited_model, success, loss, steps = self.edit_select(model, node_idx_2flip[:idx + 1].squeeze(dim=1), flipped_label[:idx + 1].squeeze(dim=1), optimizer, max_num_step, manner)
            #get success
            success = self.success_rate(model, node_idx_2flip[:idx+1].squeeze(dim=1), flipped_label[:idx+1].squeeze(dim=1))
            res = [*self.test(edited_model, whole_data), success, steps]
            hop_res = []
            for n_hop in range(1, num_htop+1):
                hop_res.append(self.get_khop_neighbors_acc(model, n_hop, idx))
            res.append(hop_res)
            results_temporary.append(res)
        return results_temporary


    def get_khop_neighbors_acc(self, model, num_hop, node_idx):
        neighbors, _, pos, _ = k_hop_subgraph(node_idx, num_hops=num_hop, edge_index=self.whole_data.edge_index)
        out = self.prediction(model, self.whole_data)
        mask = torch.ones_like(neighbors, dtype=torch.bool)
        mask[pos] = False
        neighbors = neighbors[mask]
        acc = self.compute_micro_f1(out, self.whole_data.y, neighbors)
        return acc


    def eval_edit_quality(self, node_idx_2flip, flipped_label, whole_data, max_num_step, bef_edit_results, eval_setting, manner='GD',  mixup_training_samples_idx = None, mixup_label = None):
        bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = bef_edit_results
        bef_edit_hop_acc = {}
        N_HOP = 3
        success_rate = 0
        success_list = []
        hop_drawdown = {}
        average_dd = []
        highest_dd = []
        lowest_dd = []
        test_dd_std = 0
        table_result = []
        start_time = time.time()
        #pdb.set_trace()
        for n_hop in range(1, N_HOP + 1):
            bef_edit_hop_acc[n_hop] = []
            for idx in node_idx_2flip:
                bef_edit_hop_acc[n_hop].append(self.get_khop_neighbors_acc(self.model, n_hop, idx))
        assert eval_setting in ['sequential', 'independent', 'batch']
        if eval_setting == 'sequential':
            results_temporary = self.sequential_edit(node_idx_2flip, flipped_label, whole_data, max_num_step, manner)
            train_acc, val_acc, test_acc, succeses, steps = zip(*results_temporary)
            tra_drawdown = bef_edit_tra_acc - train_acc[-1]
            val_drawdown = bef_edit_val_acc - val_acc[-1]
            test_drawdown = test_drawdown = np.round((np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals = 3).tolist()
            average_dd = np.round(np.mean(np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals=3) * 100
            test_drawdown = [test_drawdown * 100] if not isinstance(test_drawdown, list) else [round(d * 100, 1) for d in test_drawdown]
            test_dd_std = np.std(test_drawdown)
            highest_dd = max(enumerate(test_drawdown), key=lambda x: x[1])
            lowest_dd = min(enumerate(test_drawdown), key=lambda x: x[1])
            tra_std = None
            val_std = None
            test_std = None
            #pdb.set_trace()

            success_rate = np.round(np.mean(succeses), decimals = 3).tolist()
            success_list = np.round(np.array(succeses), decimals = 3).tolist()
            table_result = {
                '1st': (test_drawdown[0], success_list[0]),
                '10th': (test_drawdown[9], success_list[9]),
                '25th': (test_drawdown[24], success_list[24]),
                '50th': (test_drawdown[49], success_list[49])
            }
            hop_drawdown = {}
        elif eval_setting == 'independent' :
            results_temporary = self.independent_edit(node_idx_2flip, flipped_label, whole_data, max_num_step, num_htop=N_HOP, manner=manner)
            train_acc, val_acc, test_acc, succeses, steps, hop_acc = zip(*results_temporary)
            hop_acc = np.vstack(hop_acc)
            tra_drawdown = bef_edit_tra_acc - np.mean(train_acc)
            val_drawdown = bef_edit_val_acc - np.mean(val_acc)
            test_drawdown = bef_edit_tst_acc - np.mean(test_acc)
            tra_std = np.std(train_acc)
            val_std = np.std(val_acc)
            test_std = np.std(test_acc)
            success_rate = np.mean(succeses)
            hop_drawdown = {}
            for n_hop in range(1, N_HOP + 1):
                hop_drawdown[n_hop] = {
                                    f'{n_hop}_pre_edit_acc' : np.round(np.mean(bef_edit_hop_acc[n_hop]), decimals=3) * 100,
                                    f'{n_hop}_hops_DD': np.round(np.mean(bef_edit_hop_acc[n_hop] - hop_acc[:, n_hop-1]), decimals=3) * 100
                                    }
            # pdb.set_trace()
        elif eval_setting == 'batch':
            results_temporary = self.batch_edit(node_idx_2flip, flipped_label, whole_data, max_num_step, num_htop=0, manner=manner, mixup_training_samples_idx = mixup_training_samples_idx, mixup_label = mixup_label)
            train_acc, val_acc, test_acc, succeses, steps, hop_acc = zip(*results_temporary)
            #pdb.set_trace()
            hop_acc = np.vstack(hop_acc)
            tra_drawdown = bef_edit_tra_acc - train_acc[-1]
            val_drawdown = bef_edit_val_acc - val_acc[-1]
            test_drawdown = test_drawdown = np.round((np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals = 3).tolist()
            test_dd_std = np.std(test_drawdown)
            tra_std = None
            val_std = None
            test_std = None
            #ipdb.set_trace()

            success_rate = np.round(np.mean(succeses), decimals = 3).tolist()
            success_list = np.round(np.array(succeses), decimals = 3).tolist()

            hop_drawdown = {}
            # for n_hop in range(1, N_HOP + 1):
            #     hop_drawdown[n_hop] = {f'mean_{n_hop}_hops_DD' : np.mean(bef_edit_hop_acc[n_hop] - hop_acc[:, n_hop-1]) * 100,
            #                         f'1st_edit_{n_hop}_hops_DD': (bef_edit_hop_acc[n_hop][0] - hop_acc[0, n_hop-1]),
            #                         f'10th_edit_{n_hop}_hops_DD': (bef_edit_hop_acc[n_hop][9] - hop_acc[9, n_hop-1]),
            #                         f'25th_edit_{n_hop}_hops_DD': (bef_edit_hop_acc[n_hop][24] - hop_acc[24, n_hop-1]),
            #                         f'50th_edit_{n_hop}_hops_DD': (bef_edit_hop_acc[n_hop][49] - hop_acc[49, n_hop-1])}

            average_dd = np.round(np.mean(np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals=3) * 100
            test_drawdown = [test_drawdown * 100] if not isinstance(test_drawdown, list) else [round(d * 100, 1) for d in test_drawdown]
            test_dd_std = np.std(test_drawdown)
            highest_dd = max(enumerate(test_drawdown), key=lambda x: x[1])
            lowest_dd = min(enumerate(test_drawdown), key=lambda x: x[1])
            table_result = {
                '1': (test_drawdown[0], success_list[0]),
                '10': (test_drawdown[9], success_list[9]),
                '25': (test_drawdown[24], success_list[24]),
                '50': (test_drawdown[49], success_list[49])
            }
            if len(test_acc) > 100:
                 idx = [i*100 for i in range(len(test_acc) // 100)]
                 dd = []
                 sc = []
                 for i in idx:
                     dd.append(test_drawdown[i])
                     sc.append(succeses[i])
                 average_dd = list(zip(dd, sc))
        else:
            raise NotImplementedError
        total_time = time.time() - start_time
        return dict(bef_edit_tra_acc=bef_edit_tra_acc,
                    bef_edit_val_acc=bef_edit_val_acc,
                    bef_edit_tst_acc=bef_edit_tst_acc,
                    tra_drawdown=tra_drawdown * 100,
                    val_drawdown=val_drawdown * 100,
                    test_drawdown=test_drawdown,
                    success_rate=success_rate,
                    success_list = success_list,
                    average_dd = average_dd,
                    test_dd_std=test_dd_std,
                    highest_dd = highest_dd,
                    lowest_dd = lowest_dd,
                    total_time = total_time,
                    table_result = table_result,
                    mean_complexity=np.mean(steps),
                    hop_drawdown=hop_drawdown,
                    tra_std=tra_std,
                    val_std=val_std,
                    test_std=test_std,
                    )

    def eval_edit_generalization_quality(self, node_idx_2flip, flipped_label, whole_data, max_num_step, bef_edit_results,
                                         bef_edit_cs_results, specific_class, eval_setting, manner='GD'):
        bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = bef_edit_results
        bef_edit_cs_tra_acc, bef_edit_cs_val_acc, bef_edit_cs_tst_acc = bef_edit_cs_results
        bef_edit_hop_acc = {}
        N_HOP = 3
        for n_hop in range(1, N_HOP + 1):
            bef_edit_hop_acc[n_hop] = []
            for idx in node_idx_2flip:
                bef_edit_hop_acc[n_hop].append(self.get_khop_neighbors_acc(self.model, 1, idx))
        assert eval_setting in ['sequential', 'independent', 'batch']
        if eval_setting == 'sequential':
            results_temporary = self.sequential_edit(node_idx_2flip, flipped_label, whole_data, max_num_step, manner,
                                                     specific_class=specific_class)
            train_acc, val_acc, test_acc, cs_train_acc, cs_val_acc, cs_test_acc, succeses, steps = zip(*results_temporary)
            tra_drawdown = bef_edit_tra_acc - train_acc[-1]
            val_drawdown = bef_edit_val_acc - val_acc[-1]
            test_drawdown = np.array([bef_edit_tst_acc] * len(test_acc)) - test_acc[-1]
            cs_tra_drawdown = bef_edit_cs_tra_acc - cs_train_acc[-1]
            cs_val_drawdown = bef_edit_cs_val_acc - cs_val_acc[-1]
            cs_test_drawdown = bef_edit_cs_tst_acc - cs_test_acc[-1]
            success_rate = succeses[-1]
            hop_drawdown = {}
        elif eval_setting == 'independent' :
            results_temporary = self.independent_edit(node_idx_2flip, flipped_label, whole_data, max_num_step,
                                                      num_htop=N_HOP, manner=manner, specific_class=specific_class)
            train_acc, val_acc, test_acc, cs_train_acc, cs_val_acc, cs_test_acc, succeses, steps, hop_acc = zip(*results_temporary)
            hop_acc = np.vstack(hop_acc)
            tra_drawdown = bef_edit_tra_acc - np.mean(train_acc)
            val_drawdown = bef_edit_val_acc - np.mean(val_acc)
            test_drawdown = bef_edit_tst_acc - np.mean(test_acc)
            cs_tra_drawdown = bef_edit_cs_tra_acc - np.mean(cs_train_acc)
            cs_val_drawdown = bef_edit_cs_val_acc - np.mean(cs_val_acc)
            cs_test_drawdown = bef_edit_cs_tst_acc - np.mean(cs_test_acc)
            success_rate = np.mean(succeses)
            hop_drawdown = {}
            for n_hop in range(1, N_HOP + 1):
                hop_drawdown[n_hop] = np.mean(bef_edit_hop_acc[n_hop] - hop_acc[:, n_hop-1]) * 100
        elif eval_setting == 'batch':
            train_acc, val_acc, test_acc, succeses, steps = self.batch_edit(node_idx_2flip, flipped_label, whole_data, max_num_step, manner)
            tra_drawdown = bef_edit_tra_acc - train_acc
            val_drawdown = bef_edit_val_acc - val_acc
            test_drawdown = bef_edit_tst_acc - test_acc
            success_rate=succeses,
            if isinstance(steps, int):
                steps = [steps]
            hop_drawdown = {}
        else:
            raise NotImplementedError
        return dict(bef_edit_tra_acc=bef_edit_tra_acc,
                    bef_edit_val_acc=bef_edit_val_acc,
                    bef_edit_tst_acc=bef_edit_tst_acc,
                    bef_edit_cs_tra_acc=bef_edit_cs_tra_acc,
                    bef_edit_cs_val_acc=bef_edit_cs_val_acc,
                    bef_edit_cs_tst_acc=bef_edit_cs_tst_acc,
                    cs_train_drawdown=cs_tra_drawdown * 100,
                    cs_val_drawdown=cs_val_drawdown * 100,
                    cs_test_drawdown=cs_test_drawdown * 100,
                    tra_drawdown=tra_drawdown * 100,
                    val_drawdown=val_drawdown * 100,
                    test_drawdown=test_drawdown * 100,
                    success_rate=success_rate,
                    mean_complexity=np.mean(steps),
                    hop_drawdown=hop_drawdown,
                    )


    def grab_input(self, data: Data, indices=None):
        return {"x": data.x}

    def between_edit_finetune_mlp(self, batch_size, iters, idx, random_sampling=False):
        pass

    def success_rate(self, model, idx, label):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        model.eval()
        input = self.grab_input(self.whole_data)
        input['x'] = input['x'][idx]
        out = model(**input)
        y_pred = out.argmax(dim=-1)
        success = int(y_pred.eq(label).sum()) / label.size(0)
        torch.cuda.synchronize()
        #print(f'I am here {idx}')
        return success

class WholeGraphTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 model: BaseModel,
                 train_data: Data,
                 whole_data: Data,
                 model_config: Dict,
                 output_dir: str,
                 dataset_name: str,
                 is_multi_label_task: bool,
                 amp_mode: bool = False,
                 load_pretrained_backbone: bool = False) -> None:
        super(WholeGraphTrainer, self).__init__(
            model=model,
            train_data=train_data,
            whole_data=whole_data,
            model_config=model_config,
            output_dir=output_dir,
            dataset_name=dataset_name,
            is_multi_label_task=is_multi_label_task,
            amp_mode=amp_mode,
            load_pretrained_backbone=load_pretrained_backbone,
            args=args)


    def grab_input(self, data: Data):
        x = data.x
        i = 1
        xs = [x]
        # for SIGN
        while hasattr(data, f'x{i}'):
            xs.append(getattr(data, f'x{i}'))
            i += 1
        return {"x": data.x, 'adj_t': data.adj_t}

    def success_rate(self, model, idx, label):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        model.eval()
        input = self.grab_input(self.whole_data)
        if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GCN2_MLP', 'GAT_MLP', 'JKNET_MLP', 'GIN_MLP']:
            out = model.fast_forward(input['x'][idx], idx)
            y_pred = out.argmax(dim=-1)
        else:
            out = model(**input)
            y_pred = out.argmax(dim=-1)[idx]
        success = int(y_pred.eq(label).sum()) / label.size(0)
        torch.cuda.synchronize()
        #print(f'I am here {idx}')
        return success


    def single_edit(self, model, idx, label, optimizer, max_num_step, time_to_full_edit=False,num_edit_targets=1, pure_egnn_edit=False):
        s = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        for step in range(1, max_num_step + 1):
            optimizer.zero_grad()
            input = self.grab_input(self.whole_data)
            if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GCN2_MLP', 'GAT_MLP', 'JKNET_MLP', 'GIN_MLP']:
                if self.full_edit and time_to_full_edit:
                    model.freeze_module()
                    model.freeze_layer(model.MLP, freeze=False)
                    out = model(**input)[idx]
                    model.mlp_freezed = False
                    out += model.MLP(input['x'][idx])
                    loss = self.loss_op(out, label)
                    y_pred = out.argmax(dim=-1)
                elif pure_egnn_edit:
                    self.model.freeze_module(train=True)
                    out = model(**input)
                    loss = self.loss_op(out[idx], label)
                    y_pred = out.argmax(dim=-1)[idx]
                else:
                    out = model.fast_forward(input['x'][idx], idx)
                    loss = self.loss_op(out, label)
                    y_pred = out.argmax(dim=-1)
            else:
                out = model(**input)
                loss = self.loss_op(out[idx], label)
                y_pred = out.argmax(dim=-1)[idx]
            loss.backward()
            optimizer.step()
            if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GCN2_MLP', 'GAT_MLP', 'JKNET_MLP', 'GIN_MLP']:
                model.freeze_module(train=False)
            # sequential or independent setting
            if label.shape[0] == 1:
                if y_pred == label:
                    success = True
                    break
                else:
                    success = False
            # batch setting
            else:
                if self.stop_full_edit:
                    #pdb.set_trace()
                    success = int(y_pred[:num_edit_targets].eq(label[:num_edit_targets])[:num_edit_targets].sum()) / num_edit_targets
                elif self.stop_edit_only:
                    success = 1.0 if y_pred[num_edit_targets - 1] == label[num_edit_targets - 1] else 0
                else:
                    success = int(y_pred.eq(label).sum()) / label.size(0)
                if success == 1.:
                    if self.iters_before_stop == 0:
                        break
                    else:
                        self.iters_before_stop -= 1
        torch.cuda.synchronize()
        e = time.time()
        print(f'max allocated mem: {torch.cuda.max__allocated() / (1024**2)} MB')
        print(f'edit time: {e - s}')
        return model, success, loss, step


    def reset_mlp(self):
        for lin in self.model.MLP.lins:
            lin.weight.data.zero_()
            lin.bias.data.zero_()

    def finetune_mlp(self, batch_size, iters):
        input = self.grab_input(self.train_data)
        self.model.eval()
        # get the original GNN output embedding
        self.model.mlp_freezed = True
        with torch.no_grad():
            gnn_output = self.model(**input)
            self.model.gnn_output = self.model(**self.grab_input(self.whole_data)).cpu()
            log_gnn_output = F.log_softmax(gnn_output, dim=-1)
        # here we enable the MLP to be trained
        self.model.freeze_module(train=False)
        opt = self.get_optimizer(self.model_config, self.model)
        print('start finetuning MLP')
        s = time.time()
        torch.cuda.synchronize()
        for i in tqdm(range(iters)):
            opt.zero_grad()
            idx = np.random.choice(self.train_data.num_nodes, batch_size)
            idx = torch.from_numpy(idx).to(gnn_output.device)
            MLP_output = self.model.MLP(self.train_data.x[idx])
            # MLP_output = self.model.MLP(self.model.convs._cached_x[idx])
            cur_batch_gnn_output = gnn_output[idx]
            log_prob = F.log_softmax(MLP_output + cur_batch_gnn_output, dim=-1)
            main_loss = F.cross_entropy(MLP_output + gnn_output[idx], self.train_data.y[idx])
            kl_loss = F.kl_div(log_prob, log_gnn_output[idx], log_target=True, reduction='batchmean')
            # import ipdb; ipdb.set_trace()
            (kl_loss + main_loss).backward()
            opt.step()
        torch.cuda.synchronize()
        e = time.time()
        print(f'fine tune MLP used: {e - s} sec.')

    def between_edit_finetune_mlp(self, batch_size, iters, idx, random_sampling=False):
        input = self.grab_input(self.whole_data)
        if random_sampling:
            idx, labels = self.select_mixup_training_nodes(self.whole_data, 'wrong2correct', num_samples = batch_size)
        idx = idx.to(input['x'].device)
        #input['x'] = input['x'][idx]
        #pdb.set_trace()
        self.model.eval()
        # get the original GNN output embedding
        self.model.mlp_freezed = True
        with torch.no_grad():
            gnn_output = self.model(**input)
            log_gnn_output = F.log_softmax(gnn_output, dim=-1)
        # here we enable the MLP to be trained
        self.model.freeze_module(train=False)
        opt = self.get_optimizer(self.model_config, self.model)
        #print('start finetuning MLP between editing')
        s = time.time()
        #pdb.set_trace()
        torch.cuda.synchronize()
        for i in tqdm(range(iters)):
            opt.zero_grad()
            MLP_output = self.model.MLP(input['x'][idx])
            # MLP_output = self.model.MLP(self.model.convs._cached_x[idx])
            cur_batch_gnn_output = gnn_output[idx]
            log_prob = F.log_softmax(MLP_output + cur_batch_gnn_output, dim=-1)
            main_loss = F.cross_entropy(MLP_output + gnn_output[idx], self.whole_data.y[idx])
            kl_loss = F.kl_div(log_prob, log_gnn_output[idx], log_target=True, reduction='batchmean')
            # import ipdb; ipdb.set_trace()
            (kl_loss + main_loss).backward()
            opt.step()
        torch.cuda.synchronize()
        e = time.time()
        # print(f'fine tune MLP used: {e - s} sec.')
