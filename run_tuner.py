import argparse
import copy
import os.path as osp
from kd.configs import config as C
from kd.data.dataset import build_dataset
from kd.tuner import Tuner


# def update_cfg_by_args(cfg, args):
#     cfg.n_trials = args.n_trials
#     cfg.version = args.version

def adjust_cfg(exp_cfg, tuner_cfg):
    cfg = copy.deepcopy(exp_cfg)
    if tuner_cfg.get('gpu', None) is not None:
        cfg.trainer.gpu = tuner_cfg.gpu
    cfg.trainer.ckpt_dir = None
    # cfg.trainer.verbose = False
    return cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-mn', '--model_name', type=str, default='KDMLP', help='require `mn`.yaml and `mn_tuner.yaml`')
    # cfg_path = f'{args.model_name}.yaml'
    # tuner_cfg_path = f'{args.model_name}_tuner.yaml'
    parser.add_argument('-c', '--cfg_path', type=str, default='KDMLP.yaml')
    parser.add_argument('-tc', '--tuner_cfg_path', type=str, default='KDMLP_tuner.yaml')
    parser.add_argument('-nc', '--new_cfg_list', type=str, nargs='*', default=None)

    args = parser.parse_args()
    cfg_path = args.cfg_path # f'{args.model_name}.yaml'
    tuner_cfg_path = args.tuner_cfg_path # f'{args.model_name}_tuner.yaml'
    model_cfg = C.load_config(osp.join('./examples', cfg_path))
    tuner_cfg = C.load_config(osp.join('./examples', tuner_cfg_path))
    new_config = C.load_toml_new_cfg(args.new_cfg_list)
    C.update_config(model_cfg, new_config.get('base', {}))
    C.update_config(model_cfg, new_config.get('tuner', {}))
    exp_cfg = C.fill_dataset_cfg(model_cfg)
    exp_cfg = adjust_cfg(exp_cfg, tuner_cfg)

    dataset = build_dataset(exp_cfg.meta.dataset_name)
    tuner = Tuner(exp_cfg, tuner_cfg, dataset=dataset)
    tuner.tune()
    tuner.save_study()
    tuner.print_study_analysis()

# python run_experiments.py -c KDMLP.yaml 
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml > tmp.out
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'" "base.trainer.mask='soft'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='PubMed'" "base.trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'" "base.trainer.mask='soft'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='CiteSeer'" "base.trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'" "base.trainer.mask='soft'"