# -*- coding: utf-8 -*-
from os import path, mkdir, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
import cv2
from tqdm import tqdm

pred_folders = [
        ('dpn_softmax_f0', 'dpn_softmax_f0_test', 1),
        ('densenet_oof_pred_2', 'densenet_test_pred_2', 1),
        ('dpn_sigm_f0', 'dpn_sigm_f0_test', 1),
        ('inception_oof_pred_4', 'inception_test_pred_4', 2),
        ('oof_resnet152', 'pred_resnet152', 1),
        ('oof_resnet101_full_masks', 'pred_resnet101_full_masks', 1),
        ('oof_densenet169_softmax', 'pred_densenet169_softmax', 1),
        ('resnet_softmax', 'resnet_softmax_test', 1),
        ]

out_folder = path.join('..', 'predictions')
train_out = path.join(out_folder, 'merged_oof')
train_extend_out = path.join(out_folder, 'merged_extend_oof')

if __name__ == '__main__':
    t0 = timeit.default_timer()

    if not path.isdir(train_out):
        mkdir(train_out)
    if not path.isdir(train_extend_out):
        mkdir(train_extend_out)
        
    w_sum = np.sum([p[2] for p in pred_folders])
    
    for f in tqdm(sorted(listdir(path.join(out_folder, pred_folders[0][0])))):
        if path.isfile(path.join(out_folder, pred_folders[0][0], f)) and '.png' in f:
            pred_res = None
            ext_res = None
            for i in range(len(pred_folders)):
                pred = cv2.imread(path.join(out_folder, pred_folders[i][0], f), cv2.IMREAD_UNCHANGED).astype('float32')
                if pred_res is None:
                    pred_res = np.zeros_like(pred)
                    ext_res = np.zeros_like(pred)
                if i in [2, 5]:
                    ext_res[..., 0] += pred[..., 0]
                if i == 2:
                    pred = pred[..., ::-1]
                pred *= pred_folders[i][2]
                pred_res += pred
            pred_res /= w_sum
            ext_res /= 2
            pred_res = pred_res.astype('uint8')
            ext_res = ext_res.astype('uint8')
            cv2.imwrite(path.join(train_out, f), pred_res, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(path.join(train_extend_out, f), ext_res, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))