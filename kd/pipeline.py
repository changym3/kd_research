import os.path as osp
from kd.configs import prepare_experiment_cfg, load_config
from kd.experiment import Experiment
from kd.knowledge import extract_and_save_knowledge
from kd.data.dataset import build_dataset


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_cfg = prepare_experiment_cfg(load_config(cfg.meta.teacher), cfg.meta.dataset_name)
        self.s_cfg = prepare_experiment_cfg(load_config(cfg.meta.student), cfg.meta.dataset_name)
        self.tuner_cfg = load_config(cfg.meta.tuner)
        self.dataset = build_dataset(cfg.meta.dataset_name)

        self.ckpt_dir = osp.join(cfg.meta.pipeline_root, 'ckpt', cfg.meta.version)
        self.study_dir = osp.join(cfg.meta.pipeline_root, 'study', cfg.meta.version)
        self.gpu = cfg.meta.gpu
        self.stages = cfg.meta.stages
    
    def train_teacher(self, cfg, dataset):
        expt = Experiment(cfg, dataset=dataset)
        expt.run()

    def train_student(self, cfg, dataset, n_runs):
        expt = Experiment(cfg, dataset=dataset, n_runs=n_runs)
        expt.run()

    def sync_cfg(self, ckpt_dir, gpu):
        self.t_cfg.trainer.ckpt_dir = ckpt_dir
        self.s_cfg.trainer.kd.knowledge_dir = ckpt_dir
        self.t_cfg.trainer.gpu = gpu
        self.s_cfg.trainer.gpu = gpu
        self.t_cfg.trainer.verbose = False
        self.s_cfg.trainer.verbose = False
        # if osp.realpath(t_cfg.trainer.ckpt_dir) != osp.realpath(s_cfg.trainer.kd.knowledge_dir):
        #     print('`ckpt_dir` in teacher_cfg should be same with the `knowledge_dir` in student_cfg')
        # if t_cfg.trainer.gpu != s_cfg.trainer.gpu:
        #     print('gpu_id is not same for teacher and student')

    def run(self):
        self.sync_cfg(self.ckpt_dir, self.gpu)

        if self.stages == 'T':
            self.train_teacher(self.t_cfg, self.dataset)
        
        elif self.stages == 'TS':
            self.train_teacher(self.t_cfg, self.dataset)
            extract_and_save_knowledge(self.ckpt_dir, self.dataset)
            self.train_student(self.s_cfg, self.dataset, self.cfg.student.n_runs)

        elif self.stages == 'TT':
            self.train_teacher(self.t_cfg, self.dataset)