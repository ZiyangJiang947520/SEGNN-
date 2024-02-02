from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d, Linear
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv
from .base import BaseGNNModel
from .mlp import MLP

class GCN2(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float, theta: float = None, dropout: float = 0.0,
                 shared_weights: bool = False, batch_norm: bool = False, residual: bool = False, use_linear=False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(GCN2, self).__init__(in_channels, hidden_channels, out_channels,
                                    num_layers, dropout, batch_norm, residual, use_linear)
        self.alpha, self.theta = alpha, theta

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            if theta is None:
                conv = GCN2Conv(hidden_channels, alpha=alpha, theta=None,
                                layer=None, shared_weights=shared_weights,
                                normalize=False, add_self_loops=False)
            else:
                conv = GCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                                layer=i+1, shared_weights=shared_weights,
                                normalize=False, add_self_loops=False)
            self.convs.append(conv)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()


    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        x = x0 = self.activation(self.lins[0](x))
        x = self.dropout(x)
        for idx, conv in enumerate(self.convs[:-1]):
            h = conv(x, x0, adj_t)
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual:
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)

        h = self.convs[-1](x, x0, adj_t)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = self.activation(h)
        x = self.dropout(x)
        x = self.lins[1](x)
        return x.log_softmax(dim=-1)


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            x = x_0 = self.activation(self.lins[0](x))
            state['x_0'] = x_0[:adj_t.size(0)]
        x = self.dropout(x)
        h = self.convs[layer](x, state['x_0'], adj_t)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        x = self.activation(h)
        if layer == self.num_layers - 1:
            x = self.dropout(x)
            x = self.lins[1](x)
        return h

class GCN2_MLP(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float, theta: float = None,
                 shared_weights: bool = False, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(GCN2_MLP, self).__init__(in_channels, hidden_channels, out_channels,
                                  num_layers, dropout, batch_norm, residual)
        self.alpha, self.theta = alpha, theta

        if load_pretrained_backbone:
            self.GCN2 = GCN2.from_pretrained(
                                in_channels=in_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                saved_ckpt_path=saved_ckpt_path,
                                num_layers=num_layers,
                                alpha=alpha,
                                theta=theta,
                                dropout=dropout,
                                batch_norm=batch_norm,
                                residual=residual)
        else:
            self.GCN2 = GCN2(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,alpha=alpha, theta=theta,
                            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm, residual=residual)
        self.MLP = MLP(in_channels=in_channels, hidden_channels=hidden_channels,
                        out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                        batch_norm=batch_norm, residual=residual)

        self.mlp_freezed = True
        if load_pretrained_backbone:
            self.freeze_layer(self.GCN2, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)
        self.gnn_output = None


    def reset_parameters(self):
        ### reset GCN parameters
        for conv in self.GCN2.convs:
            conv.reset_parameters()
        if self.GCN2.batch_norm:
            for bn in self.GCN2.bns:
                bn.reset_parameters()
        for lin in self.GCN2.lins:
            lin.reset_parameters()

        ### reset MLP parameters
        for lin in self.MLP.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()

    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze

    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.GCN2, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.GCN2, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        GCN2_out = self.GCN2(x, adj_t, *args)
        if self.mlp_freezed:
            x = GCN2_out
        else:
            MLP_out = self.MLP(x, *args)
            x = GCN2_out + MLP_out
        return x

    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + self.MLP(x)
