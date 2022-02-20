import argparse
import os.path as osp
from kd.experiment import Experiment
from kd.configs import config as C

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('-c', '--cfg_path', type=str, default='GCN.yaml')
    parser.add_argument('-nc', '--new_cfg_list', type=str, nargs='*', default=None)

    args = parser.parse_args()

    model_cfg = C.load_config(osp.join('./examples', args.cfg_path))
    new_config = C.load_toml_new_cfg(args.new_cfg_list)
    C.update_config(model_cfg, new_config)
    exp_cfg= C.fill_dataset_cfg(model_cfg)

    expt = Experiment(exp_cfg)
    expt.run()


# python run_experiments.py -nc "trainer.gpu=1" "trainer.ckpt_dir='./examples/ckpt/test_GCN'"


