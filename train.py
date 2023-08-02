import argparse
import ipdb
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
parser.add_argument('--runs', default=1, type=int,
                    help='number of runs')
parser.add_argument('--output_dir', default='./ckpts', type=str)
parser.add_argument('--attack', action='store_true')
parser.add_argument('--attack_class', type=int, default=0, 
                    help='the class of nodes to be attacked')
parser.add_argument('--attack_ratio', type=float, default=0.1, 
                    help='the ratio of attacked nodes')


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
    # SIGN is special. It requries adding features before training the model.
    if model_config['arch_name'] in ['SIGN']:
        sign_transform = True
        sign_k = model_config['architecture']['num_layers']
    else:
        sign_transform = False
        sign_k = None
    data, num_features, num_classes = get_data(args.root, args.dataset, sign_transform=sign_transform, sign_k=sign_k)
    model = MODEL_FAMILY(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    model.cuda()
    print(model)
    if model_config['arch_name'] in ['SGC']:
        to_inductive = False
    else:
        to_inductive = True
    train_data, whole_data = prepare_dataset(model_config, data, args, remove_edge_index=True, inductive=to_inductive)
    del data
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    TRAINER_CLS = BaseTrainer if  model_config['arch_name'] == 'MLP' else WholeGraphTrainer
    trainer = TRAINER_CLS(args, model, train_data, whole_data, model_config, 
                          args.output_dir, args.dataset, multi_label, 
                          False)

    trainer.train()