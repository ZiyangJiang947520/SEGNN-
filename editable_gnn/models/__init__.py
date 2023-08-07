from .gcn import GCN
from .sage import SAGE
from .gcn2 import GCN2
from .mlp import MLP
from.sgc import SGC, SGC_MLP
from .sign import SIGN, SIGN_MLP

from .gcn_mlp import GCN_MLP
from .sage_mlp import SAGE_MLP


__all__ = [
    'GCN',
    'SAGE',
    'GCN2',
    'MLP',
    'GCN_MLP',
    'SAGE_MLP',
    'SGC',
    'SIGN',
    'SGC_MLP',
    'SIGN_MLP'
]