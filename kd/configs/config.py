import json
import os
import os.path as osp
import copy
from typing import Mapping
from easydict import EasyDict
import toml
import yaml


def load_config(cfg_path):
    cfg_path = os.path.realpath(cfg_path)
    with open(cfg_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        #  Loader=yaml.CLoader)
    return config

def load_toml_new_cfg(s):
    if isinstance(s, list):
        return EasyDict(toml.loads("\n".join(s)))
    else:
        return EasyDict()

def fill_dataset_cfg(model_cfg):
    cfg = copy.deepcopy(model_cfg)
    dataset = model_cfg.meta.dataset_name
    dataset_cfg_path = osp.join(osp.dirname(__file__), 'dataset_config.yaml')
    dataset_cfg = load_config(dataset_cfg_path)[dataset]
    cfg.dataset = dataset_cfg
    return cfg

def update_config(cfg, new_cfg, inplace=True):
    if inplace:
        _update_config_inplace(cfg, new_cfg)
    else:
        res_cfg = copy.deepcopy(cfg)
        _update_config_inplace(res_cfg, new_cfg)
        return res_cfg

def _update_config_inplace(cfg, new_cfg, __path=''):
    for k in new_cfg:
        if k not in cfg:
            print(f'Invalid config option: {__path[1:]}.{k}')
            continue
        elif isinstance(cfg[k], Mapping) and isinstance(new_cfg[k], Mapping):
            _update_config_inplace(cfg[k], new_cfg[k], '.'.join([__path, str(k)]))
        else:
            cfg[k] = new_cfg[k]

def print_cfg(cfg):
    print(json.dumps(cfg, indent=2))



def get_default_config(model, dataset):
    config = EasyDict()
    config_dir = os.path.dirname(__file__)
    dataset_params = load_config(os.path.join(config_dir, 'defaults/dataset_config.yaml'))
    model_params = load_config(os.path.join(config_dir, 'defaults/model_config.yaml'))
    trainer_params = load_config(os.path.join(config_dir, 'defaults/trainer_config.yaml'))
    config['meta'] = {
        'model_name': model,
        'dataset_name': dataset
    }
    config['dataset'] = dataset_params.get(dataset, None)
    config['model'] = model_params.get(model, None)
    config['trainer'] = trainer_params
    return config

def build_config(cfg_path):
    config = load_config(cfg_path)
    base_config = get_default_config(model=config.meta.model_name, dataset=config.meta.dataset_name)
    update_config(base_config, config)
    return base_config
