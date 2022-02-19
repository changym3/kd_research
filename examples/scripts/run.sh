
python run_experiments.py --model_cfg examples/GAT.yaml --dataset Cora
python run_extract_knowledge.py --ckpt_dir ./examples/ckpt/test_GAT
python run_experiments.py --model_cfg examples/KDMLP.yaml --dataset Cora --n_runs 3
# python examples/KDMLP_tuner.py --model_cfg examples/KDMLP.yaml --dataset Cora --n_trial 100 
