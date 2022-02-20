import os.path as osp
from kd.configs import fill_dataset_cfg, load_config
from kd.experiment import Experiment
from kd.knowledge import extract_and_save_knowledge
from kd.data.dataset import build_dataset
from kd.tuner import Tuner


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_cfg = fill_dataset_cfg(load_config(cfg.meta.teacher))
        self.s_cfg = fill_dataset_cfg(load_config(cfg.meta.student))
        self.ckpt_dir = osp.join(cfg.meta.pipeline_root, 'ckpt', cfg.meta.version)
        self.study_path = osp.join(cfg.meta.pipeline_root, 'study', f'{cfg.meta.version}.study')
        self.gpu = cfg.meta.gpu
        self.stages = cfg.meta.stages

        self.sync_cfg(self.ckpt_dir, self.gpu)
        self.tuner_cfg = load_config(cfg.meta.tuner)
        self.dataset = build_dataset(cfg.meta.dataset_name)
        self.tuner = Tuner(self.s_cfg, self.tuner_cfg, dataset=self.dataset)

    
    def train_teacher(self, cfg, dataset):
        expt = Experiment(cfg, dataset=dataset)
        expt.run()

    def train_student(self, cfg, dataset):
        expt = Experiment(cfg, dataset=dataset)
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

        if self.stages == 'T':
            self.train_teacher(self.t_cfg, self.dataset)
            extract_and_save_knowledge(self.ckpt_dir, self.dataset)
        elif self.stages == 'S':
            self.train_student(self.s_cfg, self.dataset)
        elif self.stages == 'Tu':
            self.tuner.tune()
            self.tuner.save_study(self.study_path)
        elif self.stages == 'TTu':
            self.train_teacher(self.t_cfg, self.dataset)
            extract_and_save_knowledge(self.ckpt_dir, self.dataset, self.t_cfg.meta.model_name)
            self.tuner.tune()
            self.tuner.save_study(self.study_path)
        # elif self.stages == 'TS':
        #     self.train_teacher(self.t_cfg, self.dataset)
        #     extract_and_save_knowledge(self.ckpt_dir, self.dataset)
        #     self.train_student(self.s_cfg, self.dataset, self.cfg.student.n_runs)

        # elif self.stages == 'Tu':
        #     self.train_teacher(self.t_cfg, self.dataset)