from typing import Tuple, Union, Optional
import time
import numpy as np
import os
import copy
import torch
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F  # 添加这个导入
import random
from collections import Counter

#============================G-mixup
from Gutils import stat_graph, split_class_graphs, align_graphs, two_graphons_mixup
from graphon_estimator import universal_svd

def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = int(y.max()) + 1

    # train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    # val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        # train_mask.scatter_(0, train_idx, True)
        train_mask[train_idx] = True
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        # val_mask.scatter_(0, val_idx, True)
        val_mask[val_idx] = True

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask


def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def get_planetoid(root: str, name: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.NormalizeFeatures(), T.SIGN(sign_K), 
                            T.RandomNodeSplit('train_rest', num_val=500, num_test=500)])
    else:
        transform = T.Compose([T.NormalizeFeatures(),
                            T.RandomNodeSplit('train_rest', num_val=500, num_test=500)])
    dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_wikics(root: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = WikiCS(f'{root}/WIKICS', transform=transform)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = Coauthor(f'{root}/Coauthor', name, transform=transform)
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = Amazon(f'{root}/Amazon', name, transform=transform)
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB', transform=transform)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB', transform=transform)
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = Yelp(f'{root}/YELP', transform=transform)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = Flickr(f'{root}/Flickr', transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    if sign_transform:
        transform = T.Compose([T.SIGN(sign_K)])
    else:
        transform = None
    dataset = Reddit2(f'{root}/Reddit2', transform=transform)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes



def get_sbm(root: str, name: str, sign_transform: bool, sign_K: int) -> Tuple[Data, int, int]:
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train')
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes


def get_data(root: str, name: str, sign_transform: bool, sign_k: int) -> Tuple[Data, int, int]:
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name, sign_transform, sign_k)
    elif name.lower() in ['coauthorcs', 'coauthorphysics']:
        return get_coauthor(root, name[8:], sign_transform, sign_k)
    elif name.lower() in ['amazoncomputers', 'amazonphoto']:
        return get_amazon(root, name[6:], sign_transform, sign_k)
    elif name.lower() == 'wikics':
        return get_wikics(root, sign_transform, sign_k)
    elif name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name, sign_transform, sign_k)
    elif name.lower() == 'reddit2':
        return get_reddit(root, sign_transform, sign_k)
    elif name.lower() == 'flickr':
        return get_flickr(root, sign_transform, sign_k)
    elif name.lower() == 'yelp':
        return get_yelp(root, sign_transform, sign_k)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root, sign_transform, sign_k)
    elif name.lower() in ['ogbn-products', 'products']:
        return get_products(root, sign_transform, sign_k)
    else:
        raise NotImplementedError


def to_inductive(data):
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    # for SIGN
    i = 1
    while hasattr(data, f'x{i}'):
        data[f'x{i}'] = data[f'x{i}'][mask]
        i += 1
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def preprocess_data(model_config, data):
    loop, normalize = model_config['loop'], model_config['normalize']
    if loop:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if normalize:
        t = time.perf_counter()
        data.adj_t = gcn_norm(data.adj_t)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')


def attack(train_data, data, attack_class, attack_ratio, save_dir, arch_name):
    # Create a mask to identify the data points with the given attack_class in the training set
    mask = train_data.y == attack_class

    # Calculate the number of data points to flip the label
    num_to_flip = int(mask.sum() * attack_ratio)
    attack_indices = torch.where(mask)[0]

    # Convert the attack_indices in the training set to indices in the whole dataset
    whole_set_indices = torch.where(data.train_mask)[0]
    attack_indices_whole_set = whole_set_indices[attack_indices]

    # Randomly select indices of the data points to flip the label
    indices_to_flip = np.random.choice(attack_indices_whole_set, size=num_to_flip, replace=False)
    
    if save_dir:
        np.save(os.path.join(save_dir, f'{arch_name}_attack_indices.npy'), indices_to_flip)
    # Flip the label of the selected data points in the training set
    for index in indices_to_flip:
        index_in_train_data = np.where(whole_set_indices == index)[0][0]
        train_data.y[index_in_train_data] = attack_class + 1
    return train_data


def prepare_dataset(model_config, data, args, remove_edge_index=True, inductive=True):
    if inductive:
        train_data = to_inductive(data)
    else:
        train_data = copy.deepcopy(data)
    if hasattr(args, 'attack') and args.attack:
        train_data = attack(train_data, data, args.attack_class, args.attack_ratio, 
                            os.path.join(args.output_dir, args.dataset), model_config['arch_name'])
    train_data = T.ToSparseTensor(remove_edge_index=remove_edge_index)(train_data.to('cuda'))
    data = T.ToSparseTensor(remove_edge_index=remove_edge_index)(data.to('cuda'))
    preprocess_data(model_config, train_data)
    preprocess_data(model_config, data)
    return train_data, data


# =========================================以下是给G-mixup的

def prepare_dataset_onehot_y(dataset):
    y_set = set()
    all_labels = []  # 用于存储所有标签
    for data in dataset:
        if data.y.dim() == 0:
            y_set.add(int(data.y.item()))  # 使用 item() 将单元素张量转换为标量
            all_labels.append(int(data.y.item()))  # 添加到标签列表
        else:
            y_set.update(data.y.tolist())  # 将多元素张量转换为列表并添加到集合中
            all_labels.extend(data.y.tolist())  # 添加到标签列表

    num_classes = max(all_labels) + 1  # 确保 num_classes 是标签中的最大值加 1
    print(f"All labels: {all_labels}")  # 打印所有标签
    print(f"Number of classes: {num_classes}")  # 打印类别数

    for data in dataset:
        if data.y.dim() == 0:
            data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
        else:
            data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float).sum(dim=0)
    return dataset







def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            data.num_nodes = int(torch.max(data.edge_index)) + 1

        if max_degree < 2000:
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ((degs - mean) / std).view(-1, 1)
    return dataset

def generate_gmixup_data(dataset, aug_ratio, aug_num, lam_range, seed):
    class_graphs = split_class_graphs(dataset)
    print(f"Class graphs: {len(class_graphs)} classes found")
    graphons = []
    for label, graphs in class_graphs.items():
        print(f"Processing class: {label}, number of graphs: {len(graphs)}")
        align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True)
        graphon = universal_svd(align_graphs_list, threshold=0.2)
        graphons.append((label, graphon))

    if len(graphons) < 2:
        raise ValueError("Not enough graphons to perform mixup. Ensure there are at least 2 classes of graphs.")

    num_sample = int(len(dataset) * aug_ratio / aug_num)
    lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

    random.seed(seed)
    new_graphs = []
    for lam in lam_list:
        two_graphons = random.sample(graphons, 2)
        new_graphs += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)

    return new_graphs





def prepare_gmixup_dataset(dataset, aug_ratio, aug_num, lam_range, seed):
    dataset = prepare_dataset_onehot_y(dataset)
    dataset = prepare_dataset_x(dataset)

    # 获取节点级别的标签
    labels = [int(label) for label in dataset[0].y]
    label_counts = Counter(labels)
    print("Label distribution in the dataset:", label_counts)

    if len(label_counts) < 2:
        raise ValueError("The dataset must contain at least two classes for mixup.")

    new_graphs = generate_gmixup_data(dataset, aug_ratio, aug_num, lam_range, seed)
    dataset = new_graphs + dataset
    return dataset


def preprocess_labels(dataset):
    for data in dataset:
        if data.y.numel() > 1:
            # 对于包含多个元素的标签，选择第一个元素作为标量标签
            data.y = data.y[0].view(1)
    return dataset
