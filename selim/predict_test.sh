#!/usr/bin/env bash
########## Resnet101 full masks for extension #############
python pred_test.py \
--gpu 0 \
--preprocessing_function caffe \
--network resnet101_2 \
--test_folder /home/selim/kaggle/datasets/bowl/stage1_test/ \
--out_masks_folder pred_resnet101_full_masks \
--out_channels 2 \
--models_dir nn_models \
--models best_resnet101_2_fold0.h5 best_resnet101_2_fold1.h5 best_resnet101_2_fold2.h5 best_resnet101_2_fold3.h5


########## InceptionResnetV2 sigmoid 2 channels #############
python pred_test.py \
--gpu 0 \
--preprocessing_function tf \
--network resnetv2 \
--test_folder /home/selim/kaggle/datasets/bowl/stage1_test/ \
--out_masks_folder pred_resnetv2fpn \
--out_channels 2 \
--models_dir nn_models \
--models best_resnetv2_fold0.h5 best_resnetv2_fold1.h5 best_resnetv2_fold2.h5 best_resnetv2_fold3.h5


########## Densenet169 softmax 2 channels #############
python pred_test.py \
--gpu 0 \
--preprocessing_function torch \
--network densenet169_softmax \
--out_masks_folder pred_densenet169_softmax \
--out_channels 3 \
--models_dir nn_models \
--models best_densenet169_softmax_fold0.h5 best_densenet169_softmax_fold1.h5 best_densenet169_softmax_fold2.h5 best_densenet169_softmax_fold3.h5


########## Resnet152 sigmoid 2 channels #############
python pred_test.py \
--gpu 0 \
--preprocessing_function caffe \
--network resnet152_2 \
--out_masks_folder pred_resnet152 \
--out_channels 2 \
--models best_resnet152_2_fold0.h5 best_resnet152_2_fold1.h5 best_resnet152_2_fold2.h5 best_resnet152_2_fold3.h5