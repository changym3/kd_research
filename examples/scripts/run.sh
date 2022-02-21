# 注意几点：
# - 如toml的字符串变量应当加上单引号''
# - 如参数名和参数值之间是等号=，而非空格
# - 每个参数周围要加引号
"meta.n_runs=5"
"meta.dataset_name='Cora'"
"trainer.gpu=1"
"trainer.ckpt_dir='./examples/ckpt/test_GCN'" 

python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=0" "meta.dataset_name='ogbn-arxiv'" "model.norm=~" "meta.n_runs=3"
python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=0" "meta.dataset_name='ogbn-arxiv'" "model.norm=bn" "meta.n_runs=3"
python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=0" "meta.dataset_name='ogbn-arxiv'" "model.num_layers=3" "meta.n_runs=3"
python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=0" "meta.dataset_name='ogbn-arxiv'" "trainer.weight_decay=1e-4" "meta.n_runs=3"


python run_experiments.py --model_cfg examples/GAT.yaml --dataset Cora


python run_experiments.py --model_cfg examples/GAT.yaml --dataset CiteSeer
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/test_CiteSeer

python run_experiments.py --model_cfg examples/GAT.yaml --dataset PubMed
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/test_PubMed

python run_pipeline.py --dataset Cora --version Cora --stages T
python run_pipeline.py --dataset Cora --version Cora --stages S
python run_pipeline.py --dataset CiteSeer --version CiteSeer --stages T
python run_pipeline.py --dataset CiteSeer --version CiteSeer --stages S
python run_pipeline.py --dataset PubMed --version PubMed --stages T
python run_pipeline.py --dataset PubMed --version PubMed --stages S


# python run_experiments.py --model_cfg examples/KDMLP.yaml --dataset Cora --n_runs 3
# python examples/KDMLP_tuner.py --model_cfg examples/KDMLP.yaml --dataset Cora --n_trial 100 