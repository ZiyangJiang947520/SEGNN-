import ipdb
import torch.nn.functional as F
import time
import torch
import numpy as np
from tqdm import tqdm
from .trainer import BaseTrainer, WholeGraphTrainer
from torch_geometric.data.data import Data
from typing import Dict
from .algs import ENN
from .utils import safe_backward


class BaseEditor(BaseTrainer):
    def __init__(self, args, model: ENN, train_data: Data, whole_data: Data, 
                 model_config: Dict, output_dir: str, dataset_name: str, is_multi_label_task: bool, 
                 amp_mode: bool = False) -> None:
        super().__init__(args, model, train_data, whole_data, model_config, output_dir, 
                         dataset_name, is_multi_label_task, amp_mode)
        
        self.original_model = self.model.model
        self.edit_gen = self.batch_generator()
        self.opt = self.get_optimizer(self.model_config, self.model.model)

    def run(self):
        for i in tqdm(range(self.model.config.n_epochs)):
            self.train_step()

    def train_step(self):
        batch = next(self.edit_gen)
        l_total, l_edit, l_loc = self.edit_step(batch, training=True)
        print(f'edit loss: {l_edit}, locality loss: {l_loc}')
        self.opt.step()
        self.opt.zero_grad()
                

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)
        with torch.no_grad():
            try:
                base_logits = self.model(batch['x'])
            except:
                import ipdb; ipdb.set_trace()
        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch['x'], None, batch['edit'], self.loss_op)
        edit_time = time.time() - start
        # print(edit_time)
    
        with torch.set_grad_enabled(training):
            # Editing loss
            edit_idx, edit_label = batch['edit']['idx'], batch['edit']['label']
            if len(edit_label.shape) == 2:
                edit_label = edit_label.squeeze()
            post_edit_logits = edited_model(batch['x'])
            l_edit = self.loss_op(post_edit_logits[edit_idx], edit_label)

            # Locality loss
            loc_idx = batch['loc']['idx']
            l_loc = F.kl_div(F.log_softmax(base_logits[loc_idx].detach()), F.log_softmax(post_edit_logits[loc_idx]), log_target=True)

        l_total_edit = self.model.config.cedit * l_edit + self.model.config.cloc * l_loc
        safe_backward(l_total_edit, self.model.outer_parameters())

        return l_total_edit, l_edit, l_loc


    def batch_generator(self):
        batch = self.grab_input(self.whole_data)
        while True:
            n_edits = self.model.config.n_edits
            n_locs = self.model.config.batch_size - n_edits
            node_idx_2flip, flipped_label = self.select_node(self.whole_data, 
                                                             self.original_model.out_channels, 
                                                             n_edits, 
                                                             'random', 
                                                             from_valid_set=False)
            node_idx_2flip, flipped_label = node_idx_2flip.cuda(), flipped_label.cuda()
            flip_flag = torch.rand(node_idx_2flip.shape, device=node_idx_2flip.device) > 0.5
            flip_flag = flip_flag.long()
            labels = self.whole_data.y[node_idx_2flip] * flip_flag + (1 - flip_flag) * flipped_label

            loc_idx = self.generate_loc_idx(n_locs, node_idx_2flip)
            batch["edit"] = {"idx": node_idx_2flip.squeeze(), "label": labels}
            batch["loc"] = {"idx": loc_idx.squeeze(), "label": self.whole_data.y[loc_idx]}
            yield batch


    def generate_loc_idx(self, n_locs, node_idx_2flip):
        train_node_set = self.whole_data.train_mask.nonzero().squeeze()
        perm = torch.randperm(train_node_set.size(0))
        loc_idx = train_node_set[perm[:n_locs]]
        while len(np.intersect1d(node_idx_2flip.cpu().numpy(), loc_idx.cpu().numpy())) > 0:
            perm = torch.randperm(train_node_set.size(0))
            loc_idx = train_node_set[perm[:n_locs]]
        return loc_idx


class WholeGraphEditor(WholeGraphTrainer):
    def __init__(self, args, model: ENN, train_data: Data, whole_data: Data, 
                 model_config: Dict, output_dir: str, dataset_name: str, is_multi_label_task: bool, 
                 amp_mode: bool = False) -> None:
        super().__init__(args, model, train_data, whole_data, model_config, output_dir, 
                         dataset_name, is_multi_label_task, amp_mode)
        
        self.original_model = self.model.model
        self.edit_gen = self.batch_generator()
        self.opt = self.get_optimizer(self.model_config, self.model.model)

    def run(self):
        for i in tqdm(range(self.model.config.n_epochs)):
            self.train_step()

    def train_step(self):
        batch = next(self.edit_gen)
        l_total, l_edit, l_loc = self.edit_step(batch, training=True)
        print(f'edit loss: {l_edit}, locality loss: {l_loc}')
        self.opt.step()
        self.opt.zero_grad()
                

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)
        with torch.no_grad():
            base_logits = self.model(batch['x'], batch['adj_t'])
        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch['x'], batch['adj_t'], batch['edit'], self.loss_op)
        edit_time = time.time() - start
        # print(edit_time)
    
        with torch.set_grad_enabled(training):
            # Editing loss
            edit_idx, edit_label = batch['edit']['idx'], batch['edit']['label']
            if len(edit_label.shape) == 2:
                edit_label = edit_label.squeeze()
            post_edit_logits = edited_model(batch['x'], batch['adj_t'])
            l_edit = self.loss_op(post_edit_logits[edit_idx], edit_label)

            # Locality loss
            loc_idx = batch['loc']['idx']
            l_loc = F.kl_div(F.log_softmax(base_logits[loc_idx].detach()), F.log_softmax(post_edit_logits[loc_idx]), log_target=True)

        l_total_edit = self.model.config.cedit * l_edit + self.model.config.cloc * l_loc
        safe_backward(l_total_edit, self.model.outer_parameters())

        return l_total_edit, l_edit, l_loc


    def batch_generator(self):
        batch = self.grab_input(self.whole_data)
        while True:
            n_edits = self.model.config.n_edits
            n_locs = self.model.config.batch_size - n_edits
            node_idx_2flip, flipped_label = self.select_node(self.whole_data, 
                                                             self.original_model.out_channels, 
                                                             n_edits, 
                                                             'random', 
                                                             from_valid_set=False)
            node_idx_2flip, flipped_label = node_idx_2flip.cuda(), flipped_label.cuda()
            flip_flag = torch.rand(node_idx_2flip.shape, device=node_idx_2flip.device) > 0.5
            flip_flag = flip_flag.long()
            labels = self.whole_data.y[node_idx_2flip] * flip_flag + (1 - flip_flag) * flipped_label

            loc_idx = self.generate_loc_idx(n_locs, node_idx_2flip)
            batch["edit"] = {"idx": node_idx_2flip.squeeze(), "label": labels}
            batch["loc"] = {"idx": loc_idx.squeeze(), "label": self.whole_data.y[loc_idx]}
            yield batch


    def generate_loc_idx(self, n_locs, node_idx_2flip):
        train_node_set = self.whole_data.train_mask.nonzero().squeeze()
        perm = torch.randperm(train_node_set.size(0))
        loc_idx = train_node_set[perm[:n_locs]]
        while len(np.intersect1d(node_idx_2flip.cpu().numpy(), loc_idx.cpu().numpy())) > 0:
            perm = torch.randperm(train_node_set.size(0))
            loc_idx = train_node_set[perm[:n_locs]]
        return loc_idx


def test():
    import os
    import sys
    import copy
    import torch_geometric.transforms as T
    sys.path.append(os.getcwd())
    sys.path.append('/home/zl105/edit_gnn')

    from conf import ModelConfig, EnnConfig
    from data import get_data
    import models

    dataset = 'cora'
    ROOT = '/home/zl105/data'
    config_path = '../config/gcn.yaml'
    model_config = ModelConfig.from_directory(config_path, dataset)
    data, num_features, num_classes = get_data(ROOT, dataset)
    data = T.ToSparseTensor()(data)
    data = data.to('cuda')
    MODEL_FAMILY = getattr(models, model_config['arch_name'])
    model = MODEL_FAMILY(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    print(model_config)
    config = {
        "edit_lr": 0.1,
        "n_edit_steps": 1,
        "batch_size": 32,
        "n_edits": 16, 
        "first_order": False,
        "cedit": 1.0,
        "cloc": 1.0,
        "model": {
            "inner_params": ["convs.1.weight", "convs.1.bias"]
        }
    }
    enn_config = EnnConfig.from_dict(config)
    print(enn_config)

    enn = ENN(model, enn_config, lambda: copy.deepcopy(model)).cuda()

    trainer = WholeGraphEditor(enn, data, data, model_config, '/tmp', 'cora', False, False, 1, 0)
    orig_param = [p.clone() for (n, p) in enn.model.named_parameters()]

    trainer.train_step()

    edited_param = [p for (n, p) in enn.model.named_parameters()]

    print((orig_param[0] - edited_param[0]).abs().max())


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        test()
