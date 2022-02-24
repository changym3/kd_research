# bash examples/scripts/run_batch.sh > tmp.out
# cat tmp.out | grep "Best Epoch - CiteSeer"
# CiteSeer Cora PubMed ogbn-arxiv
# soft, hidden, logit

# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=0" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=1" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=2" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=3" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=4" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=5" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=6" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=7" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_experiments.py -c KDMLP.yaml -nc "trainer.gpu=1" "model.aug.k=8" "meta.dataset_name='Test'" "trainer.kd.knowledge_dir='./examples/ckpt/Test_GCN/'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'" "base.trainer.mask=''


python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'" "base.trainer.mask='feats'"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='PubMed'" "base.trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'" "base.trainer.mask='feats'"
python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='CiteSeer'" "base.trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'" "base.trainer.mask='feats'"

# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='Cora'" "base.trainer.kd.knowledge_dir='./examples/ckpt/Cora_GCN/'" "base.trainer.mask='hidden'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='PubMed'" "base.trainer.kd.knowledge_dir='./examples/ckpt/PubMed_GCN/'" "base.trainer.mask='hidden'"
# python run_tuner.py -c KDMLP.yaml -tc KDMLP_tuner.yaml -nc "base.meta.dataset_name='CiteSeer'" "base.trainer.kd.knowledge_dir='./examples/ckpt/CiteSeer_GCN/'" "base.trainer.mask='hidden'"
