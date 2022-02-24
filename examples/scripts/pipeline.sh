# arxiv 

python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=1" "meta.n_runs=1" "trainer.ckpt_dir='./examples/ckpt/arxiv_GCN'" 
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/arxiv_GCN
<!-- The Prediction, All: 0.7542, Train: 0.7827, Val: 0.7278, Test: 0.7169 -->

python run_experiments.py -c KDMLP_arxiv.yaml -nc "trainer.gpu=0" 
<!-- Best Epoch - Test: 0.547106146812439 -->

python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='ogbn-arxiv'" "base.trainer.kd.knowledge_dir='./examples/ckpt/arxiv_GCN/'"


# Cora
python run_experiments.py -c GCN_cora.yaml -nc "trainer.gpu=1" "meta.n_runs=1" "trainer.ckpt_dir='./examples/ckpt/cora_GCN'"
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/Cora_GCN
<!-- The Prediction, All: 0.8272, Train: 0.9786, Val: 0.8060, Test: 0.8220 -->

python run_experiments.py -c KDMLP_cora.yaml -nc "trainer.gpu=0"
<!-- Best Epoch - Test: 0.8184 ± 0.0042 -->

python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'"

# Citeseer

python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/PubMed_GCN
<!-- The Prediction, All: 0.7051, Train: 0.9833, Val: 0.7280, Test: 0.7180 -->

python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='CiteSeer'" "base.trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'"

# PubMed
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/CiteSeer_GCN
<!-- The Prediction, All: 0.7824, Train: 1.0000, Val: 0.8060, Test: 0.7960 -->

python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='PubMed'" "base.trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'"


运行KDMLP
python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=0" "meta.dataset_name='PubMed'" "trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'" "trainer.ckpt_dir='./examples/ckpt/PubMed_KD/'"
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/PubMed_KD --model_name KDMLP

python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=0" "meta.dataset_name='CiteSeer'" "trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'" "trainer.ckpt_dir='./examples/ckpt/CiteSeer_KD/'"
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/CiteSeer_KD --model_name KDMLP

python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=0" "meta.dataset_name='ogbn-arxiv'" "trainer.kd.knowledge_dir='./examples/ckpt/ogbn-arxiv_GCN/'" "trainer.ckpt_dir='./examples/ckpt/ogbn-arxiv_KD/'"
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/ogbn-arxiv_KD --model_name KDMLP