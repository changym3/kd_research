'''
trainer teacher -> extract&save knowledge -> tune student

'''

import argparse
from kd.configs import load_config
from kd.pipeline import Pipeline


def update_cfg_by_args(cfg, args):
    cfg.meta.dataset_name = args.dataset
    cfg.meta.version = args.version
    cfg.meta.stages = args.stages



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_cfg', type=str, default='./examples/pipeline.yaml')
    # parser.add_argument('--teacher_cfg', type=str, default='./examples/GAT.yaml')
    # parser.add_argument('--student_cfg', type=str, default='./examples/KDMLP.yaml')
    # parser.add_argument('--tuner_cfg', type=str, default='./examples/KDMLP_tuner.yaml')
    # parser.add_argument('--gpu', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--version', type=str, default='test_pipeline')
    parser.add_argument('--stages', type=str, default='S', choices=['T', 'S', 'Tu', 'TTu'], help='T=teacher, S=student, Tu=tune')

    args = parser.parse_args()

    pl_cfg = load_config(args.pipeline_cfg)
    update_cfg_by_args(pl_cfg, args)

    pl = Pipeline(pl_cfg)
    pl.run()


# python run_pipeline.py --dataset PubMed --version PubMed --stages T
# python run_pipeline.py --dataset Cora --version Cora --stages T
# python run_pipeline.py --dataset CiteSeer --version CiteSeer --stages T