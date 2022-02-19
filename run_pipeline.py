'''
trainer teacher -> extract&save knowledge -> tune student

'''

import os.path as osp
import argparse
from kd.configs import load_config
from kd.configs import prepare_experiment_cfg
from kd.experiment import Experiment
from kd.knowledge import extract_and_save_knowledge
from kd.data.dataset import build_dataset

def train_teacher(cfg, dataset):
    expt = Experiment(cfg, dataset=dataset)
    expt.run()

def train_student(cfg, dataset, n_student_repeats):
    expt = Experiment(cfg, dataset=dataset, n_runs=n_student_repeats)
    expt.run()


def adjust_cfg(t_cfg, s_cfg):
    if osp.realpath(t_cfg.trainer.ckpt_dir) != osp.realpath(s_cfg.trainer.kd.knowledge_dir):
        print('`ckpt_dir` in teacher_cfg should be same with the `knowledge_dir` in student_cfg')
    if t_cfg.trainer.gpu != s_cfg.trainer.gpu:
        print('gpu_id is not same for teacher and student')

    t_cfg.trainer.verbose = False
    s_cfg.trainer.verbose = False


def main(t_cfg, s_cfg, args):
    dataset = build_dataset(t_cfg.meta.dataset_name)

    train_teacher(t_cfg, dataset)
    extract_and_save_knowledge(t_cfg.trainer.ckpt_dir, dataset)
    train_student(s_cfg, dataset, args.n_student_repeats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('--teacher_cfg', type=str, default='./examples/GAT.yaml')
    parser.add_argument('--student_cfg', type=str, default='./examples/KDMLP.yaml')
    parser.add_argument('--tuner_cfg', type=str, default='./examples/KDMLP_tuner.yaml')

    parser.add_argument('--dataset', type=str, default='Cora')
    
    # for consistency
    # parser.add_argument('--ckpt_dir', type=str, default='./examples/ckpt/test_GAT', help='`ckpt_dir` in teacher_cfg should be same with the `knowledge_dir` in student_cfg')
    # parser.add_argument('--gpu', type=int, default=0)

    # tune
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--n_trial_repeats', type=int, default=1)

    # student
    parser.add_argument('--n_student_repeats', type=int, default=1)

    args = parser.parse_args()

    t_cfg = prepare_experiment_cfg(load_config(args.teacher_cfg), args.dataset)
    s_cfg = prepare_experiment_cfg(load_config(args.student_cfg), args.dataset)
    adjust_cfg(t_cfg, s_cfg)
    
    main(t_cfg, s_cfg, args)


# python run_pipeline.py --dataset PubMed