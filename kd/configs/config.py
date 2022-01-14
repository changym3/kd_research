import os
import copy
from easydict import EasyDict
import yaml


def get_default_config(model, dataset):
    config = EasyDict()
    config_dir = os.path.dirname(__file__)
    dataset_params = EasyDict(yaml.load(open(os.path.join(config_dir, 'defaults/dataset_config.yaml')), Loader=yaml.FullLoader))
    model_params = EasyDict(yaml.load(open(os.path.join(config_dir, 'defaults/model_config.yaml')), Loader=yaml.FullLoader))
    trainer_params = EasyDict(yaml.load(open(os.path.join(config_dir, 'defaults/trainer_config.yaml')), Loader=yaml.FullLoader))
    
    config['meta'] = {
        'model_name': model,
        'dataset_name': dataset
    }
    config['dataset'] = dataset_params[dataset]
    config['model'] = model_params[model]
    config['trainer'] = trainer_params
    return config


def build_config(cfg_path):
    cfg_path = os.path.realpath(cfg_path)
    config = EasyDict(yaml.load(open(cfg_path), Loader=yaml.FullLoader))
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