import os
import yaml
import dataclasses
from dataclasses import asdict, dataclass, field

@dataclass
class ModelConfig:
    architecture: dict = field(default=None)
    optim: str = field(default='adam')
    lr: float = field(default=0.01)
    epochs: int = field(default=None)
    arch_name: str = field(default=None)
    norm: bool = field(default=None)
    loop: bool = field(default=None)


    @classmethod
    def from_directory(cls, config_path, dataset):
        if not config_path:
            raise ValueError("undefined config_path.")
        with open(config_path, 'r') as fp:
            model_config = yaml.load(fp, Loader=yaml.FullLoader)
            loop = model_config.get('loop', False)
            normalize = model_config.get('norm', False)
            if dataset == 'reddit2':
                model_config = model_config['params']['reddit']
            else:
                model_config = model_config['params'][dataset]
            model_config['loop'] = loop
            model_config['normalize'] = normalize
        return model_config


    @classmethod
    def from_dict(cls, info_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in info_dict.items() if k in field_names})


    def write_to_directory(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f)


@dataclass
class EnnConfig:
    edit_lr: float = field(default=0.1)
    n_edit_steps: int = field(default=2)
    n_edits: int = field(default=16)
    first_order: bool = field(default=False)
    model: dict = field(default=None)
    batch_size: int = field(default=32)
    cedit: float = field(default=1.0)
    cloc: float = field(default=1.0)
    n_epochs: int = field(default=100)


    @classmethod
    def from_dict(cls, info_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in info_dict.items() if k in field_names})
    

    @classmethod
    def from_directory(cls, config_path):
        if not config_path:
            raise ValueError("undefined config_path.")
        with open(config_path, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        return cls.from_dict(config)
