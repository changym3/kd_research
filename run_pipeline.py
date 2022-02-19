'''
trainer teacher -> extract&save knowledge -> tune student

'''

import argparse
from kd.configs import load_config
from kd.pipeline import Pipeline


def update_cfg_by_args(cfg, args):
    cfg.meta.dataset_name = args.dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_cfg', type=str, default='./examples/pipeline.yaml')
    # parser.add_argument('--teacher_cfg', type=str, default='./examples/GAT.yaml')
    # parser.add_argument('--student_cfg', type=str, default='./examples/KDMLP.yaml')
    # parser.add_argument('--tuner_cfg', type=str, default='./examples/KDMLP_tuner.yaml')

    parser.add_argument('--stages', type=str, default='TS', choices=['T', 'TS', 'TT', 'TTS'], help='T=teacher, TS=teacher+student, TT=teacher+tune, TTS=teacher+tune+student')
    parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)

    args = parser.parse_args()

    pl_cfg = load_config(args.pipeline_cfg)
    update_cfg_by_args(pl_cfg, args)
    print(pl_cfg)

    pl = Pipeline(pl_cfg)
    pl.run()


# python run_pipeline.py --dataset PubMed