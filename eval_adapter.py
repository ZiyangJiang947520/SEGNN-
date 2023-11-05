import argparse
import torch
import os
import numpy as np
import pdb
import json
import torch.nn.functional as F
import yaml
import editable_gnn.models as models
from data import get_data, prepare_dataset
from editable_gnn import WholeGraphTrainer, BaseTrainer, set_seeds_all
from editable_gnn.models import GCN, MLP
from editable_gnn.models.base import BaseGNNModel


class GCN_Adapter(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False):
        super(GCN_Adapter, self).__init__(in_channels, hidden_channels, out_channels, \
                                  num_layers, dropout, batch_norm, residual)
        # self.alpha, self.theta = alpha, theta

        self.GCN = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,\
                        num_layers=num_layers, dropout=dropout, batch_norm=batch_norm, residual=residual)
        self.adapter = torch.nn.ModuleList([MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
                        out_channels=hidden_channels, num_layers=2, dropout=dropout,
                        batch_norm=batch_norm, residual=residual) for _ in range(num_layers)])
        
        self.mlp_freezed = True
        self.freeze_module(train=True)
        self.gnn_output = None

    def reset_parameters(self):
        ### reset GCN parameters
        for conv in self.GCN.convs:
            conv.reset_parameters()
        if self.GCN.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

        ### reset MLP parameters
        for mlp in self.adapter:
            for lin in mlp.lins:
                lin.reset_parameters()
            if mlp.batch_norm:
                for bn in mlp.bns:
                    bn.reset_parameters()
        
    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze
            
    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.GCN, freeze=False)
            self.freeze_layer(self.adapter, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.GCN, freeze=True)
            self.freeze_layer(self.adapter, freeze=False)
            self.mlp_freezed = False



    def forward_wo_adapter(self, x, adj_t, *args, **kwargs):
        for idx in range(self.GCN.num_layers - 1):
            conv = self.GCN.convs[idx]
            h = conv(x, adj_t)
            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            # import ipdb; ipdb.set_trace()
            x = self.dropout(x)
        h = self.GCN.convs[-1](x, adj_t, *args)
        if self.use_linear:
            linear = self.lins[-1](x)
            x = h + linear
        else:
            x = h
        return x
    
    def forward(self, x, adj_t, *args, **kwargs):
        for idx in range(self.GCN.num_layers - 1):
            conv = self.GCN.convs[idx]
            h = conv(x, adj_t)
            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            # import ipdb; ipdb.set_trace()
            x = self.adapter[idx](x)
            x = self.dropout(x)
        h = self.GCN.convs[-1](x, adj_t, *args)
        if self.use_linear:
            linear = self.lins[-1](x)
            x = h + linear
        else:
            x = h
        return x

    @classmethod
    def from_existing(cls, model, **kwargs):
        new_model = cls(**kwargs)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.GCN.named_parameters()):
            assert n1 == n2
            p2.data.copy_(p1.data)
        return new_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--saved_model_path', type=str, required=True,
                    help='the path to the traiend model')
parser.add_argument('--output_dir', default='./finetune', type=str)
parser.add_argument('--num_samples', default=50, type=int)
parser.add_argument('--runs', default=1, type=int,
                    help='number of runs')
parser.add_argument('--criterion', type=str, required=True, help='the criterion of how to select the node need to be patched.' \
                                                                  'currently only support ``wrong_to_correct`` and ``random``')
parser.add_argument('--manner', type=str, required=True, default='GD', \
                    choices=['GD','GD_Diff', 'Ada_GD_Diff', 'EDG', 'EDG_Plus'], help='edit manner for finetuning')
parser.add_argument('--hyper_Diff', default=1.0, type=float, help='the hyperparameter for Diff loss')
parser.add_argument('--train_split', default=1, type=int, help='Training data split number for EDG_Plus')
parser.add_argument('--gamma', default=1.0, type=float, help='the hyperparameter for adapative Diff loss')

MAX_NUM_EDIT_STEPS = 200
MAX_NUM_EDIT_STEPS_FOR_BATCH = 200


if __name__ == '__main__':
    args = parser.parse_args()
    set_seeds_all(args.seed)
    with open(args.config, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        if args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        else:
            model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize
    print(args)
    print(f'Edit manner: {args.manner}')
    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    MODEL_FAMILY = getattr(models, model_config['arch_name'])
    if model_config['arch_name'] in ['SIGN', 'SIGN_MLP']:
        sign_transform = True
        sign_k = model_config['architecture']['num_layers']
    else:
        sign_transform = False
        sign_k = None
    data, num_features, num_classes = get_data(args.root, args.dataset, sign_transform=sign_transform, sign_k=sign_k)
    # print(f'data={data}')
    model = MODEL_FAMILY.from_pretrained(in_channels=num_features, 
                                out_channels=num_classes, 
                                saved_ckpt_path=args.saved_model_path,
                                **model_config['architecture'])

    model_with_adapter = GCN_Adapter.from_existing(model, 
                                                   in_channels=num_features,
                                                   out_channels=num_classes, 
                                                   **model_config['architecture'])
    model_with_adapter.cuda()
    model.cuda()
    model_with_adapter.freeze_module(train=False)
    for n, p in model_with_adapter.GCN.named_parameters():
        assert p.requires_grad == False

    if model_config['arch_name'] in ['SGC', 'SGC_MLP']:
        to_inductive = False
    else:
        to_inductive = True
    train_data, whole_data = prepare_dataset(model_config, data, args, remove_edge_index=False, inductive=to_inductive)
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    TRAINER_CLS = BaseTrainer if model_config['arch_name'] == 'MLP' else WholeGraphTrainer
    trainer = TRAINER_CLS(args=args,
                          model=model_with_adapter, 
                          train_data=train_data, 
                          whole_data=whole_data, 
                          model_config=model_config, 
                          output_dir=args.output_dir, 
                          dataset_name=args.dataset, 
                          is_multi_label_task=multi_label, 
                          amp_mode=False)
    


    bef_edit_results = trainer.test(model, whole_data)
    train_acc, valid_acc, test_acc = bef_edit_results
    print(f'before edit, train acc {train_acc}, valid acc {valid_acc}, test acc {test_acc}')
    

    model_with_adapter.freeze_module(train=False) ### train MLP module and freeze GNN module
    MAX_NUM_EDIT_STEPS = 500
    MAX_NUM_EDIT_STEPS_FOR_BATCH = 500
    # IMPORTANT NOTE: here we found that batch size is crusial for fine-tuning GCN_MLP on reddit2! Large batch size matters.
    if args.dataset == 'flickr' or (args.dataset == 'reddit2' and model_config['arch_name'] == 'GCN_MLP') or \
        (args.dataset in ['amazoncomputers', 'amazonphoto', 'coauthorcs', 'coauthorphysics']):
        trainer.finetune_adapter(batch_size=512, iters=100)
    else:
        trainer.finetune_adapter(batch_size=32, iters=100)
    bef_edit_ft_results = trainer.test(model, whole_data)
    ft_train_acc, ft_valid_acc, ft_test_acc = bef_edit_ft_results
    print(f'before edit, after fine tune, train acc {ft_train_acc}, valid acc {ft_valid_acc}, test acc {ft_test_acc}')

    assert args.criterion in ['wrong2correct', 'random'], 'currently only support selecting nodes with mode ' \
                                                          '``wrong2correct`` or ``random``'
    node_idx_2flip, flipped_label = trainer.select_node(whole_data=whole_data, 
                                                        num_classes=num_classes, 
                                                        num_samples=args.num_samples, 
                                                        criterion=args.criterion, 
                                                        from_valid_set=True)
                                                        
    node_idx_2flip, flipped_label = node_idx_2flip.cuda(), flipped_label.cuda()

    # print(f'the calculated stats averaged over {args.num_samples} sequential edit '
    #         f'max allocated steps: {MAX_NUM_EDIT_STEPS}')
    # seq_results = trainer.eval_edit_quality(node_idx_2flip=node_idx_2flip, 
    #                                     flipped_label=flipped_label, 
    #                                     whole_data=whole_data, 
    #                                     max_num_step=MAX_NUM_EDIT_STEPS, 
    #                                     bef_edit_results=bef_edit_results, 
    #                                     eval_setting='sequential',
    #                                     manner=args.manner)
    # print(seq_results)

    print(f'the calculated stats after {args.num_samples} independent edit '
            f'max allocated steps: {MAX_NUM_EDIT_STEPS}')
    ind_results = trainer.eval_edit_quality(node_idx_2flip=node_idx_2flip, 
                                        flipped_label=flipped_label, 
                                        whole_data=whole_data, 
                                        max_num_step=MAX_NUM_EDIT_STEPS, 
                                        bef_edit_results=bef_edit_results, 
                                        eval_setting='independent',
                                        manner=args.manner)
    print(ind_results)

    # print(f'the calculated stats after batch edit with batch size {args.num_samples}, '
    #         f'max allocated steps: {MAX_NUM_EDIT_STEPS_FOR_BATCH}')
    # batch_results = trainer.eval_edit_quality(node_idx_2flip=node_idx_2flip, 
    #                                     flipped_label=flipped_label, 
    #                                     whole_data=whole_data, 
    #                                     max_num_step=MAX_NUM_EDIT_STEPS_FOR_BATCH, 
    #                                     bef_edit_results=bef_edit_results, 
    #                                     eval_setting='batch',
    #                                     manner=args.manner)
    # print(batch_results)
    summary = {
               'ind_edit': ind_results, 
               'model_config': model_config}
    root_json = f'{args.output_dir}/{args.dataset}/{args.manner}/'  
    if not os.path.exists(root_json):
        os.makedirs(root_json)
    if args.manner == 'GD':
        json_name = root_json + f'{MODEL_FAMILY.__name__}_{args.criterion}_eval.json'
    elif args.manner == 'GD_Diff':
        json_name = root_json + f'{MODEL_FAMILY.__name__}_{args.criterion}_eval_hyper_Diff={args.hyper_Diff}.json'
    elif args.manner == 'Ada_GD_Diff':
        json_name = root_json + f'{MODEL_FAMILY.__name__}_{args.criterion}_eval_hyper_Diff={args.hyper_Diff}.json'
    elif args.manner == 'EDG':
        json_name = root_json + f'{MODEL_FAMILY.__name__}_{args.criterion}_eval_gamma={args.gamma}.json'
    else:
        json_name = root_json + f'{MODEL_FAMILY.__name__}_{args.criterion}_eval_train_split={args.train_split}_gamma={args.gamma}.json'

    with open(json_name, 'w') as f:
        json.dump(summary, f)