import argparse
import torch
import ipdb
import shutil
import copy
import numpy as np
import random
import pdb
import json
import torch.nn.functional as F
import yaml
import editable_gnn.models as models
from data import get_data, prepare_dataset
from editable_gnn import set_seeds_all, WholeGraphEditor, BaseEditor, EnnConfig, ENN


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--alg_config', type=str, required=True, 
                    help='the path to the alg configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--saved_model_path', type=str, required=True,
                    help='the path to the traiend model')
parser.add_argument('--output_dir', default='./finetune', type=str)
parser.add_argument('--runs', default=1, type=int,
                    help='number of runs')


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
    enn_config = EnnConfig.from_directory(args.alg_config)
    print(args)
    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    MODEL_FAMILY = getattr(models, model_config['arch_name'])
    data, num_features, num_classes = get_data(args.root, args.dataset)
    model = MODEL_FAMILY.from_pretrained(in_channels=num_features, 
                                out_channels=num_classes, 
                                saved_ckpt_path=args.saved_model_path,
                                **model_config['architecture'])
    print(model)
    model.cuda()
    enn = ENN(model, enn_config, lambda: copy.deepcopy(model)).cuda()
    train_data, whole_data = prepare_dataset(model_config, data, args, remove_edge_index=False)
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    # TRAINER_CLS = BaseTrainer if model_config['arch_name'] == 'MLP' else WholeGraphEditor
    TRAINER_CLS = BaseEditor if model_config['arch_name'] == 'MLP' else WholeGraphEditor
    trainer = TRAINER_CLS(args=args,
                          model=enn, 
                          train_data=train_data, 
                          whole_data=whole_data, 
                          model_config=model_config, 
                          output_dir=args.output_dir, 
                          dataset_name=args.dataset, 
                          is_multi_label_task=multi_label, 
                          amp_mode=False)
    trainer.run()

    trainer.save_model(f"{trainer.model_name}", enn.config.n_epochs)