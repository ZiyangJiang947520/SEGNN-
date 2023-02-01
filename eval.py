import argparse
import torch
import shutil
import numpy as np
import random
import pdb
import json
import torch.nn.functional as F
import yaml
import editable_gnn.models as models
from data import get_data, prepare_dataset
from editable_gnn import WholeGraphTrainer, BaseTrainer, set_seeds_all


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
parser.add_argument('--criterion', type=str, required=True, help='the criterion of how to select the node need to be patched.' \
                                                                  'currently only support ``wrong_to_correct`` and ``random``')

MAX_NUM_EDIT_STEPS = 10
MAX_NUM_EDIT_STEPS_FOR_BATCH = 20


def edit(model, data, optimizer, loss_op, node_idx_2flip, flipped_label, max_num_step):
    model.train()
    for i in range(max_num_step):
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = loss_op(out[node_idx_2flip].unsqueeze(0), flipped_label)
        loss.backward()
        optimizer.step()
        y_pred = out.argmax(dim=-1)[node_idx_2flip]
        # pdb.set_trace()
        print(f'{i}-th edit step, loss: {loss.item()}, model pred: {y_pred.item()}, label: {flipped_label.item()}')
        if y_pred == flipped_label:
            print(f'successfully flip the model with {i} grad decent steps, break')
            break


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
    train_data, whole_data = prepare_dataset(model_config, data)
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    N = 1000
    TRAINER_CLS = BaseTrainer if model_config['arch_name'] == 'MLP' else WholeGraphTrainer
    trainer = TRAINER_CLS(model=model, 
                          train_data=train_data, 
                          whole_data=whole_data, 
                          model_config=model_config, 
                          output_dir=args.output_dir, 
                          dataset_name=args.dataset, 
                          is_multi_label_task=multi_label, 
                          amp_mode=False, 
                          runs=1, 
                          seed=args.seed)

    bef_edit_results = trainer.test(model, whole_data)
    train_acc, valid_acc, test_acc = bef_edit_results
    print(f'before edit, train acc {train_acc}, valid acc {valid_acc}, test acc {test_acc}')

    assert args.criterion in ['wrong2correct', 'random'], 'currently only support selecting nodes with mode ' \
                                                          '``wrong2correct`` or ``random``'
    node_idx_2flip, flipped_label = trainer.select_node(whole_data=whole_data, 
                                                        num_classes=num_classes, 
                                                        num_samples=args.num_samples, 
                                                        criterion=args.criterion, 
                                                        from_valid_set=True)
                                                        
    node_idx_2flip, flipped_label = node_idx_2flip.cuda(), flipped_label.cuda()

    print(f'the calculated stats averaged over {args.num_samples} sequential edit '
            f'max allocated steps: {MAX_NUM_EDIT_STEPS}')
    seq_results = trainer.eval_edit_quality(node_idx_2flip=node_idx_2flip, 
                                        flipped_label=flipped_label, 
                                        whole_data=whole_data, 
                                        max_num_step=MAX_NUM_EDIT_STEPS, 
                                        bef_edit_results=bef_edit_results, 
                                        eval_setting='sequential')
    print(seq_results)

    print(f'the calculated stats after {args.num_samples} independent edit '
            f'max allocated steps: {MAX_NUM_EDIT_STEPS}')
    ind_results = trainer.eval_edit_quality(node_idx_2flip=node_idx_2flip, 
                                        flipped_label=flipped_label, 
                                        whole_data=whole_data, 
                                        max_num_step=MAX_NUM_EDIT_STEPS, 
                                        bef_edit_results=bef_edit_results, 
                                        eval_setting='independent')
    print(ind_results)

    print(f'the calculated stats after batch edit with batch size {args.num_samples}, '
            f'max allocated steps: {MAX_NUM_EDIT_STEPS_FOR_BATCH}')
    batch_results = trainer.eval_edit_quality(node_idx_2flip=node_idx_2flip, 
                                        flipped_label=flipped_label, 
                                        whole_data=whole_data, 
                                        max_num_step=MAX_NUM_EDIT_STEPS_FOR_BATCH, 
                                        bef_edit_results=bef_edit_results, 
                                        eval_setting='batch')
    print(batch_results)
    summary = {'seq_edit': seq_results, 
               'ind_edit': ind_results, 
               'batch_edit': batch_results,
               'model_config': model_config}
    with open(f'{args.output_dir}/{args.dataset}/{MODEL_FAMILY.__name__}_{args.criterion}_eval.json', 'w') as f:
        json.dump(summary, f)