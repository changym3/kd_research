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

    parser.add_argument('--stages', type=str, default='TS', choices=['T', 'S', 'TS', 'TT', 'TTS'], help='T=teacher, TS=teacher+student, TT=teacher+tune, TTS=teacher+tune+student')
    parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--version', type=str, default='test_pipeline')

    args = parser.parse_args()

    pl_cfg = load_config(args.pipeline_cfg)
    update_cfg_by_args(pl_cfg, args)

    pl = Pipeline(pl_cfg)
    pl.run()


# python run_pipeline.py --dataset PubMed --version PubMed --stages T