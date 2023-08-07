import torch
from torch.nn import BatchNorm1d
from torch import Tensor
from torch_sparse import SparseTensor
from pathlib import Path
from torch.nn import Linear
from typing import List
from .mlp import MLP



class SIGN(torch.nn.Module):
    def __init__(self, in_channels: int,out_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.0):
        super(SIGN, self).__init__()
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(Linear(in_channels, hidden_channels))  
        self.lin = Linear((num_layers + 1) * hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, xs: List, *args, **kwargs) -> Tensor:
        hs = []
        assert len(xs) == len(self.lins)
        for x, lin in zip(xs, self.lins):
            h = lin(x).relu()
            h = self.dropout(h)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        h = self.lin(h)
        return h
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()


    @classmethod
    def from_pretrained(cls, in_channels: int, out_channels: int, saved_ckpt_path: str, **kwargs):
        model = cls(in_channels=in_channels, out_channels=out_channels, **kwargs)
        if not saved_ckpt_path.endswith('.pt'):
            checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__}_*.pt")]
            if '_MLP' not in cls.__name__:
                glob_checkpoints = [x for x in checkpoints if '_MLP' not in x]
            else:
                glob_checkpoints = checkpoints
            # print(checkpoints)

            # checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__}_run*.pt")]
            # glob_checkpoints = checkpoints
            assert len(glob_checkpoints) == 1
            saved_ckpt_path = glob_checkpoints[0]
        print(f'load model weights from {saved_ckpt_path}')
        state_dict = torch.load(saved_ckpt_path, map_location='cpu')
        final_state_dict = {}
        ignore_keys = ['edit_lrs']
        for k, v in state_dict.items():
            if k in ignore_keys:
                continue
            if k.startswith('model'):
                new_k = k.split('model.')[1]
                final_state_dict[new_k] = v
            else:
                final_state_dict[k] = v
        model.load_state_dict(final_state_dict, strict=False)
        return model


class SIGN_MLP(SIGN):
    def __init__(self, in_channels: int,out_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.0):
        super(SIGN_MLP, self).__init__(in_channels, out_channels, hidden_channels, num_layers, dropout)

        self.MLP = MLP(in_channels=in_channels, hidden_channels=hidden_channels,
                        out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        
        self.mlp_freezed = True
        self.freeze_module(train=True)
        self.gnn_output = None

    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze
            
    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.lin, freeze=False)
            self.freeze_layer(self.lins, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.lin, freeze=True)
            self.freeze_layer(self.lins, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False


    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx].to(x.device) + self.MLP(x)
    
    def reset_parameters(self):
        ### reset MLP parameters
        for lin in self.MLP.lins:
            lin.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()


    def forward(self, xs: List, *args, **kwargs) -> Tensor:
        hs = []
        assert len(xs) == len(self.lins)
        for x, lin in zip(xs, self.lins):
            h = lin(x).relu()
            h = self.dropout(h)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        h = self.lin(h)
        if self.mlp_freezed:
            x = h
        else:   
            MLP_out = self.MLP(xs[0])
            x = h + MLP_out
        return x