import os
import os.path as osp
import copy
from easydict import EasyDict
import yaml


def load_config(cfg_path):
    cfg_path = os.path.realpath(cfg_path)
    with open(cfg_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return config

def prepare_experiment_cfg(model_cfg, dataset):
    cfg = copy.deepcopy(model_cfg)
    dataset_cfg_path = osp.join(osp.dirname(__file__), 'dataset_config.yaml')
    dataset_cfg = load_config(dataset_cfg_path)[dataset]
    cfg.meta.dataset_name = dataset
    cfg.dataset = dataset_cfg
    return cfg


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

def update_config(cfg, new_cfg, inplace=True):
    if inplace:
        _update_config_inplace(cfg, new_cfg)
        return cfg
    else:
        res_cfg = copy.deepcopy(cfg)
        _update_config_inplace(res_cfg, new_cfg)
        return res_cfg

def _update_config_inplace(cfg, new_cfg, __path=''):
    # should assert the cfg has the same nested structure as new_cfg
    for k in new_cfg:
        # if cfg.get(k, None) is None:
            # print(f'Invalid config option: {__path[1:]}.{k}')
            # cfg[k] = v
            # continue
        if cfg.get(k, None) is None:
            cfg[k] = new_cfg[k]
        elif isinstance(cfg[k], EasyDict) and isinstance(new_cfg[k], EasyDict):
            _update_config_inplace(cfg[k], new_cfg[k], '.'.join([__path, str(k)]))
        else:
            cfg[k] = new_cfg[k]