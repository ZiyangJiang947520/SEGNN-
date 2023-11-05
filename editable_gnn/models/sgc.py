import torch
from torch.nn import BatchNorm1d
from torch import Tensor
from torch_geometric.nn import SGConv 
from torch_sparse import SparseTensor
from pathlib import Path
from .mlp import MLP



class SGC(torch.nn.Module):
    def __init__(self, in_channels: int,out_channels: int, num_layers: int):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K=num_layers, cached=True)
    
    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        x = self.conv(x, adj_t)
        return x
    
    def reset_parameters(self):
        self.conv.reset_parameters()
    

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


class SGC_MLP(SGC):
    def __init__(self, in_channels: int,out_channels: int, num_layers: int, hidden_channels: int, 
                 batch_norm: bool = False, residual: bool = False, dropout: float = 0.0):
        super(SGC_MLP, self).__init__(in_channels, out_channels, num_layers)

        self.MLP = MLP(in_channels=in_channels, hidden_channels=hidden_channels,
                        out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                        batch_norm=batch_norm, residual=residual)
        
        self.mlp_freezed = True
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
        self.conv.reset_parameters()
        ### reset MLP parameters
        for lin in self.MLP.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        GCN_out = self.conv(x, adj_t, *args)
        if self.mlp_freezed:
            x = GCN_out
        else:   
            # MLP_out = self.MLP(x, *args)
            MLP_out = self.MLP(self.conv._cached_x, *args)
            x = GCN_out + MLP_out
        return x