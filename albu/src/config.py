from collections import namedtuple

Config = namedtuple("Config", [
    "dataset_path",
    "iter_size",
    "folder",
    "target_rows",
    "target_cols",
    "num_channels",
    "network",
    "loss",
    "optimizer",
    "lr",
    "lr_steps",
    "lr_gamma",
    "batch_size",
    "epoch_size",
    "nb_epoch",
    "predict_batch_size",
    "test_pad",
    "results_dir",
    "num_classes",
    "ignore_target_size",
    "warmup",
    "sigmoid"
])


