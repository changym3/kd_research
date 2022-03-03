# bash examples/scripts/run_batch.sh > tmp.out
# cat tmp.out | grep "Best Epoch - Test"
# CiteSeer Cora PubMed ogbn-arxiv
# soft, hidden, logit

# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=0"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=1"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=2"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=3"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=4"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=5"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=6"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "model.raw.hop=7"

python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=1"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=2"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=3"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=4"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=5"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=6"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=7"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "base.model.raw.hop=8"




# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'" "base.trainer.mask='feats'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='PubMed'" "base.trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'" "base.trainer.mask='feats'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='CiteSeer'" "base.trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'" "base.trainer.mask='feats'"

# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'" "base.trainer.mask='hidden'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='PubMed'" "base.trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'" "base.trainer.mask='hidden'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='CiteSeer'" "base.trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'" "base.trainer.mask='hidden'"
