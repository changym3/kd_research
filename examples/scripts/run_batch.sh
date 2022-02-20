# bash ./scripts/run.sh > tmp.out
# cat tmp.out | grep "Best Epoch - Test"

python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=1" 
python run_experiments.py -c GCN_arxiv.yaml -nc "trainer.gpu=1" "model.norm='bn'"

Highest Train: 0.7390 ± 0.0024
Best Epoch : 476.6667 ± 21.0317
Best Epoch - Train: 0.7368 ± 0.0026
Best Epoch - Valid: 0.7160 ± 0.0008
Best Epoch - Test: 0.7065 ± 0.0025


# bash examples/scripts/run_batch.sh > tmp.out