#!/usr/bin/env bash


##################### Resnet152 FPN  with Sigmoid activation ##############################
python train.py \
--gpu "0"  \
--fold "0,1,2,3" \
--num_workers 8  \
--network resnet152_2 \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.00002  \
--decay 0.0001  \
--batch_size 16  \
--crop_size 224 \
--steps_per_epoch 500 \
--epochs 16 \
--preprocessing_function caffe \
--weights "nn_models/best_resnet152_2_fold{}.h5"


##################### Densenet169 FPN with Softmax activation ##############################

python train.py \
--gpu "0"  \
--fold "0,1,2,3" \
--num_workers 8  \
--network densenet169_softmax \
--freeze_till_layer input_1  \
--loss categorical_dice \
--optimizer adam  \
--use_softmax \
--learning_rate 0.00002  \
--decay 0.0001  \
--batch_size 16  \
--crop_size 256 \
--steps_per_epoch 500 \
--epochs 16 \
--preprocessing_function torch \
--weights "nn_models/best_densenet169_softmax_fold{}.h5"

##################### Resnet101 FPN  Full masks with Sigmoid activation ##############################

python train.py \
--gpu "0"  \
--fold "0,1,2,3" \
--num_workers 8  \
--network resnet101_2 \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.00002  \
--decay 0.0001  \
--batch_size 16  \
--crop_size 256 \
--steps_per_epoch 500 \
--epochs 16 \
--use_full_masks \
--preprocessing_function caffe \
--weights "nn_models/best_resnet101_2_fold{}.h5"

