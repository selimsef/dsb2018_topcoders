# dsb2018_topcoders
DSB2018 [ods.ai] topcoders 

## How to run predict
```bash
./predict_test.sh
```

After it prediction will be in _stub_ folder. And submission files will be in _stub_ folder. Individual model predictions will be in _predictions_ folder.

## How to run training
```bash
./train_all.sh
./tune_all.sh
./predict_oof_trees.sh
```

We use two stage training because we want to tune models on stage1 data released 11.04.
Every script goes into every folder and runs scripts to train models.
