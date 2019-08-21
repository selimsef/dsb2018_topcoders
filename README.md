
# DSB2018 [ods.ai] topcoders 1st place solution 

## Model weights/annotated data
[You can download the whole package (7.5G)](https://www.dropbox.com/s/qvtgbz0bnskn9wu/dsb2018_topcoders.zip?dl=0) that contains training data, nn models weights, GBT models.

See also [solution description on Kaggle](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741)


You need to setup your environment first. Please install latest nvidia drivers, cuda 9 and cudnn 7.
After that run setup_env.sh script

## How to run predict
unzip test data into data_test folder and
```bash
./predict_test.sh
```

Submission files will be in _predictions_ folder (submission_0.csv, submission_1.csv). 
Individual model predictions will be also in _predictions_ folder.

## How to run training
Before training please remove models from:
* albu/weights
* selim/nn_models
* victor/nn_models
* victor/lgbm_models

after it run:
```bash
./train_all.sh
./tune_all.sh
./predict_oof_trees.sh
```

We use two stage training because we want to tune models on the stage1 data released 11.04.
Every script goes into every folder and runs scripts to train models.

