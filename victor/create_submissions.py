# -*- coding: utf-8 -*-
from os import path, listdir, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
from skimage.color import label2rgb
from tqdm import tqdm
from multiprocessing import Pool
import lightgbm as lgb
from train_classifier import get_inputs
import pandas as pd

data_folder = path.join('..', 'data')
pred_folder = path.join('..', 'predictions')

test_pred_folder = path.join(pred_folder, 'merged_test')

test_images_folder = path.join('..', 'data_test')

lgbm_models_folder = 'lgbm_models'

test_out_folders = ['lgbm_test_sub1', 'lgbm_test_sub2']
color_out_folders = ['color_test_sub1', 'color_test_sub2']

extend_mask_test = path.join(pred_folder, 'merged_extend_test')

DATA_THREADS = 12

num_split_iters = 50
folds_count = 3

sep_count = 3

best_thrs = [0.3, 0.2]

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def pred_to_rles(y_pred):
    for i in range(1, y_pred.max() + 1):
        yield rle_encoding(y_pred == i)
        
            
if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    for f in test_out_folders:
        if not path.isdir(path.join(pred_folder, f)):
            mkdir(path.join(pred_folder, f))
        
    for f in color_out_folders:
        if not path.isdir(path.join(pred_folder, f)):
            mkdir(path.join(pred_folder, f))
            
    all_files = []
    inputs = []
    outputs = []
    inputs2 = []
    outputs2 = []
    labels = []
    labels2 = []
    separated_regions = []
    fns = []
    paramss = []
    
    gbm_models = []
    
    for it in range(num_split_iters):
        for it2 in range(folds_count):
            gbm_models.append(lgb.Booster(model_file=path.join(lgbm_models_folder, 'gbm_model_{0}_{1}.txt'.format(it, it2))))
    
    inputs = []
    paramss = []
    for f in tqdm(sorted(listdir(test_pred_folder))):
        if path.isfile(path.join(test_pred_folder, f)) and '.png' in f:
            img_id = f.split('.')[0]
            paramss.append((f, test_pred_folder, path.join(test_images_folder, img_id, 'images'), None, extend_mask_test))
    
    inputs = []
    inputs2 = []
    labels= []
    labels2 = []
    separated_regions= []
    with Pool(processes=DATA_THREADS) as pool:
        results = pool.starmap(get_inputs, paramss)
    for i in range(len(results)):
        inp, lbl, inp2, lbl2, sep_regs = results[i]
        inputs.append(inp)
        inputs2.append(inp2)
        labels.append(lbl)
        labels2.append(lbl2)
        separated_regions.append(sep_regs)

    for sub_id in range(2):
        
        print('Creating submission', sub_id)
        new_test_ids = []
        rles = []
    
        bst_k = np.zeros((sep_count+1))
        removed = 0
        replaced = 0
        total_cnt = 0
        im_idx = 0
        
        empty_cnt = 0
        
        for f in tqdm(sorted(listdir(test_pred_folder))):
            if path.isfile(path.join(test_pred_folder, f)) and '.png' in f:
                img_id = f.split('.')[0]
                
                inp = inputs[im_idx]
                pred = np.zeros((inp.shape[0]))
                pred2 = [np.zeros((inp2.shape[0])) for inp2 in inputs2[im_idx]]
                
                for m in gbm_models:
                    if pred.shape[0] > 0:
                        pred += m.predict(inp)
                    for k in range(len(inputs2[im_idx])):
                        if pred2[k].shape[0] > 0:
                            pred2[k] += m.predict(inputs2[im_idx][k])
                if pred.shape[0] > 0:
                    pred /= len(gbm_models)
                for k in range(len(pred2)):
                    if pred2[k].shape[0] > 0:
                        pred2[k] /= len(gbm_models)
                
                pred_labels = np.zeros_like(labels[im_idx], dtype='uint16')
                
                clr = 1
                
                for i in range(pred.shape[0]):
                    max_sep = -1
                    max_pr = pred[i]
                    for k in range(len(separated_regions[im_idx])):
                        if len(separated_regions[im_idx][k][i]) > 0:
                            pred_lvl2 = pred2[k][separated_regions[im_idx][k][i]]
                            if len(pred_lvl2) > 1 and pred_lvl2.mean() > max_pr:
                                max_sep = k
                                max_pr = pred_lvl2.mean()
                                break
                            if len(pred_lvl2) > 1 and pred_lvl2.max() > max_pr:
                                max_sep = k
                                max_pr = pred_lvl2.max()
                                
                    if max_sep >= 0:
                        pred_lvl2 = pred2[max_sep][separated_regions[im_idx][max_sep][i]]
                        replaced += 1
                        for j in separated_regions[im_idx][max_sep][i]:
                            if pred2[max_sep][j] > best_thrs[sub_id]:
                                pred_labels[labels2[im_idx][max_sep] == j+1] = clr
                                clr += 1
                            else:
                                removed += 1
                    else:
                        if pred[i] > best_thrs[sub_id]:
                            pred_labels[labels[im_idx] == i+1] = clr
                            clr += 1
                        else:
                            removed += 1
                    bst_k[max_sep+1] += 1
                    
                total_cnt += pred_labels.max()
        
                cv2.imwrite(path.join(pred_folder, test_out_folders[sub_id], f.replace('.png', '.tif')), pred_labels)
                
                rle = list(pred_to_rles(pred_labels))
                if len(rle) == 0:
                    empty_cnt += 1
                    rles.extend([''])
                    new_test_ids.extend([img_id])
                else:
                    rles.extend(rle)
                    new_test_ids.extend([img_id] * len(rle))
                
                clr_labels = label2rgb(pred_labels, bg_label=0)
                clr_labels *= 255
                clr_labels = clr_labels.astype('uint8')
                cv2.imwrite(path.join(pred_folder, color_out_folders[sub_id], '{0}.png'.format(img_id)), clr_labels, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
                im_idx += 1
    
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(path.join(pred_folder, 'submission_{0}.csv'.format(sub_id)), index=False)
    
        print('total_cnt', total_cnt, 'removed', removed, 'replaced', replaced, 'empty:', empty_cnt)
        print(bst_k)
    
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))