# GAT
python run_experiments.py --model_cfg examples/GAT.yaml --dataset Cora
python run_experiments.py --model_cfg examples/GAT.yaml --dataset CiteSeer
python run_experiments.py --model_cfg examples/GAT.yaml --dataset PubMed


# KD-GAT
python run_experiments.py --model_cfg examples/GAT.yaml --dataset Cora
python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDGAT.yaml --dataset Cora --n_runs 3

python run_experiments.py --model_cfg examples/GAT.yaml --dataset CiteSeer
python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDGAT.yaml --dataset CiteSeer --n_runs 3

python run_experiments.py --model_cfg examples/GAT.yaml --dataset PubMed
python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDGAT.yaml --dataset PubMed --n_runs 3

# KD-MLP
python run_experiments.py --model_cfg examples/GAT.yaml --dataset Cora
python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDMLP.yaml --dataset Cora --n_runs 3

python run_experiments.py --model_cfg examples/GAT.yaml --dataset CiteSeer
python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDMLP.yaml --dataset CiteSeer --n_runs 3

python run_experiments.py --model_cfg examples/GAT.yaml --dataset PubMed
python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDMLP.yaml --dataset PubMed --n_runs 3