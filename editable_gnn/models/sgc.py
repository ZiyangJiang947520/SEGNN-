import torch
from torch.nn import BatchNorm1d
from torch import Tensor
from torch_geometric.nn import SGConv 
from torch_sparse import SparseTensor
from pathlib import Path
from .mlp import MLP



class SGC(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(SGC, self).__init__(in_channels, hidden_channels, out_channels, 
                                  num_layers, dropout, batch_norm, residual, use_linear)
        self.convs.append(SGConv(in_channels, out_channels, K=num_layers, cached=True))
    
    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        for idx in range(self.num_layers - 1):
            conv = self.convs[idx]
            x = conv(x, adj_t)
        return x

class SGC_MLP(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 shared_weights: bool = True, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(SGC_MLP, self).__init__(in_channels, hidden_channels, out_channels, 
                                    num_layers, dropout, batch_norm, residual)

        if load_pretrained_backbone:
            self.SGC = SGC.from_pretrained(
                in_channels=in_channels, 
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                saved_ckpt_path=saved_ckpt_path,
                num_layers=num_layers, 
                dropout=dropout, 
                batch_norm=batch_norm, 
                residual=residual)
        else:
            self.SGC = SGC(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,\
                            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm, residual=residual)
        self.MLP = MLP(in_channels=in_channels, hidden_channels=hidden_channels,
                        out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                        batch_norm=batch_norm, residual=residual)
        
        self.mlp_freezed = True
        if load_pretrained_backbone:
            self.freeze_layer(self.SGC, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)
        self.gnn_output = None


    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze
            
    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.conv, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.conv, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False


    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx].to(x.device) + self.MLP(x)
    
    def reset_parameters(self):
        ### reset GCN parameters
        for conv in self.SGC.convs:
            conv.reset_parameters()
        
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
            self.freeze_layer(self.SGC, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.SGC, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        SGC_out = self.SGC(x, adj_t, *args)
        if self.mlp_freezed:
            x = SGC_out
        else:   
            MLP_out = self.MLP(x, *args)
            x = SGC_out + MLP_out
        return x

    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + self.MLP(x)