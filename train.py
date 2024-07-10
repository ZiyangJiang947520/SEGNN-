import argparse
import os
import ipdb
import yaml
import editable_gnn.models as models
from data import get_data, prepare_dataset
from editable_gnn import WholeGraphTrainer, BaseTrainer, set_seeds_all
from editable_gnn.utils import str2bool
import torch
from data import get_data, prepare_dataset, prepare_dataset_onehot_y, prepare_dataset_x, generate_gmixup_data, prepare_gmixup_dataset, preprocess_labels
from collections import Counter
from torch_geometric.data import Data

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
parser.add_argument('--finetune_between_edit', type=str2bool, default=False,
                        help="whether to finetune the MLP between editing")
parser.add_argument('--stop_edit_only', type=str2bool, default=False,
                        help="whether to stop when the edit target is correct")
parser.add_argument('--stop_full_edit', type=str2bool, default=False,
                        help="whether to stop when all of the edit targets are correct")
parser.add_argument('--iters_before_stop', type=int, default=0,
                        help="more iterations to run before full stopping")
parser.add_argument('--full_edit', type=int, default=0,
                        help="whether to edit both the gnn and mlp")
parser.add_argument('--pure_egnn', type=int, default=0,
                        help="whether to use pure egnn in the first iterations")
parser.add_argument('--mixup_k_nearest_neighbors', type=str2bool, default=False,
                        help="whether to sample k nearest neighbors for training mixup")
parser.add_argument('--incremental_batching', type=str2bool, default=False,
                        help="whether to do incremental batching edit")
parser.add_argument('--half_half', type=str2bool, default=False,
                        help="half and half mixup")
parser.add_argument('--half_half_ratio_mixup', type=float, default=0.5,
                        help="ratio for half and half mixup. This ratio is used as ratio of NN samples in the mixup.")
parser.add_argument('--sliding_batching', type=int, default=0,
                        help="whether to do sliding batching edit")
parser.add_argument('--num_mixup_training_samples', default=0, type=int)
parser.add_argument('--wrong_ratio_mixup', type=float, default=0.0,
                        help="ratio for wrong samples. This ratio is used as ratio of wrong samples in the mixup.")
parser.add_argument('--use_betweenness_centrality', type=str2bool, default=True,
                    help="Whether to compute and use betweenness centrality.")
parser.add_argument('--use_closeness_centrality', type=str2bool, default=True,
                    help="Whether to compute and use closeness centrality.")
parser.add_argument('--use_eigenvector_centrality', type=str2bool, default=True,
                    help="Whether to compute and use eigenvector centrality.")
parser.add_argument('--use_combined_centrality', type=str2bool, default=True,
                    help="Whether to compute and use combined (degree and betweenness) centrality.")

if __name__ == '__main__':
    args = parser.parse_args()
    set_seeds_all(args.seed)
    with open(args.config, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        load_pretrained_backbone = model_config.get('load_pretrained_backbone', False)
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
    if model_config['arch_name'] in ['SIGN', 'SIGN_MLP']:
        sign_transform = True
        sign_k = model_config['architecture']['num_layers']
    else:
        sign_transform = False
        sign_k = None
    data, num_features, num_classes = get_data(args.root, args.dataset, sign_transform=sign_transform, sign_k=sign_k)
    save_path = os.path.join(args.output_dir, args.dataset)
    model = MODEL_FAMILY(in_channels=num_features, out_channels=num_classes, load_pretrained_backbone = load_pretrained_backbone, saved_ckpt_path = save_path, **model_config['architecture'])
    model.cuda()
    print(model)
    if model_config['arch_name'] in ['SGC', 'SGC_MLP']:
        to_inductive = False
    else:
        to_inductive = True

    # # 加载数据集
    # data, num_features, num_classes = get_data(args.root, args.dataset, sign_transform=sign_transform, sign_k=sign_k)
    #
    # # 确保 data 是 Data 对象的列表
    # if isinstance(data, Data):
    #     data = [data]
    #
    # # 打印数据集信息，确保加载正确
    # print("Number of graphs in the dataset:", len(data))
    # print("Example graph:", data[0])
    #
    # # 获取节点级别的标签并打印标签分布
    # labels = [int(label) for label in data[0].y]
    # label_counts = Counter(labels)
    # print("Label distribution in the dataset:", label_counts)
    #
    # if len(label_counts) < 2:
    #     raise ValueError("The dataset must contain at least two classes for mixup.")
    #
    # # Prepare G-Mixup data
    # aug_ratio = 0.15  # Example value, adjust as necessary
    # aug_num = 10  # Example value, adjust as necessary
    # lam_range = [0.005, 0.01]  # Example value, adjust as necessary
    #
    # dataset = prepare_gmixup_dataset(data, aug_ratio, aug_num, lam_range, args.seed)
    # # Save the prepared dataset for future use
    # gmixup_data_path = os.path.join(args.output_dir, args.dataset, 'gmixup_data.pt')
    # torch.save(dataset, gmixup_data_path)
    # print(f'G-Mixup data saved to {gmixup_data_path}')

    train_data, whole_data = prepare_dataset(model_config, data, args, remove_edge_index=False, inductive=to_inductive)
    del data
    print("whole_data.edge_index:", whole_data.edge_index)
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    #ipdb.set_trace()

    if args.use_betweenness_centrality:
        betweenness_centrality = BaseTrainer.compute_betweenness_centrality(whole_data.edge_index,
                                                                            whole_data.num_nodes)
        centrality_save_path = os.path.join(args.output_dir, args.dataset, 'betweenness_centrality.pt')
        torch.save(betweenness_centrality, centrality_save_path)

    elif args.use_closeness_centrality:
        closeness_centrality = BaseTrainer.compute_closeness_centrality(whole_data.edge_index,
                                                                            whole_data.num_nodes)
        centrality_save_path = os.path.join(args.output_dir, args.dataset, 'closeness_centrality.pt')
        torch.save(closeness_centrality, centrality_save_path)

    elif args.use_eigenvector_centrality:
        eigenvector_centrality = BaseTrainer.compute_eigenvector_centrality(whole_data.edge_index,
                                                                            whole_data.num_nodes)
        centrality_save_path = os.path.join(args.output_dir, args.dataset, 'eigenvector_centrality.pt')
        torch.save(eigenvector_centrality, centrality_save_path)

    # Save the graphon model
    class_graphs = split_class_graphs(dataset[:train_nums])
    graphons = []
    for label, graphs in class_graphs:
        align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True,
                                                                                    N=resolution)
        graphon = universal_svd(align_graphs_list, threshold=0.2)
        graphons.append((label, graphon))

    graphon_save_path = os.path.join(args.output_dir, args.dataset, 'graphon_model.pt')
    torch.save(graphons, graphon_save_path)
    print(f'Graphon model saved to {graphon_save_path}')


    TRAINER_CLS = BaseTrainer if  model_config['arch_name'] == 'MLP' else WholeGraphTrainer
    trainer = TRAINER_CLS(args, model, train_data, whole_data, model_config,
                          args.output_dir, args.dataset, multi_label,
                          False,  load_pretrained_backbone=load_pretrained_backbone)

    trainer.train()
