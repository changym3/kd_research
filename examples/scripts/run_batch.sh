# bash ./scripts/run.sh > tmp.out
# cat tmp.out | grep "Best Epoch - Test"

# bash examples/scripts/run_batch.sh > tmp.out

python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=0" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=1" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=2" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=3" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=4" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=5" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=6" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=7" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=8" "meta.dataset_name='PubMed'"
python run_experiments.py -c MLP.yaml -nc "trainer.gpu=1" "model.aug.k=9" "meta.dataset_name='PubMed'"
