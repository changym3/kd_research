# arxiv 

python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=1" "meta.n_runs=1" "trainer.ckpt_dir='./examples/ckpt/arxiv_GCN'" 
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/arxiv_GCN
<!-- The Prediction, All: 0.7542, Train: 0.7827, Val: 0.7278, Test: 0.7169 -->

python run_experiments.py -c KDMLP_arxiv.yaml -nc "trainer.gpu=0" 
<!-- Best Epoch - Test: 0.547106146812439 -->


# Cora
python run_experiments.py -c GCN_cora.yaml -nc "trainer.gpu=1" "meta.n_runs=1" "trainer.ckpt_dir='./examples/ckpt/cora_GCN'"
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/cora_GCN
<!-- The Prediction, All: 0.8272, Train: 0.9786, Val: 0.8060, Test: 0.8220 -->

python run_experiments.py -c KDMLP_cora.yaml -nc "trainer.gpu=0"
<!-- Best Epoch - Test: 0.8184 ± 0.0042 -->