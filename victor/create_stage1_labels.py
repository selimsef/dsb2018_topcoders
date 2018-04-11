import sys
from os import path, mkdir, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2
import timeit
from tqdm import tqdm
from multiprocessing import Pool
import shutil

root_data_folder = path.join('..', 'data')

data_folder = path.join(root_data_folder, 'stage1_test')
masks_labels_folder = path.join(root_data_folder, 'labels_all')
images_out = path.join(root_data_folder, 'images_all')

def create_mask(img_id):
    labels = None
    i = 0
    for f in listdir(path.join(data_folder, img_id, 'masks')):
        if not path.isfile(path.join(data_folder, img_id, 'masks', f)) or '.png' not in f:
            continue
        i += 1
        msk = cv2.imread(path.join(data_folder, img_id, 'masks', f), cv2.IMREAD_GRAYSCALE)
        if labels is None:
            labels = np.zeros_like(msk, dtype='uint16')
        labels[msk > 0] = i
    
    cv2.imwrite(path.join(masks_labels_folder, '{0}.tif'.format(img_id)), labels)
    return None

if __name__ == '__main__':
    t0 = timeit.default_timer()

    if not path.isdir(masks_labels_folder):
        mkdir(masks_labels_folder)
#
    if not path.isdir(images_out):
        mkdir(images_out)
        
    paramss = []
    for d in tqdm(listdir(data_folder)):
        if not path.isdir(path.join(data_folder, d)):
            continue
        paramss.append((d,))
        shutil.copyfile(path.join(data_folder, d, 'images', '{0}.png'.format(d)), path.join(images_out, '{0}.png'.format(d)))
        
    with Pool(processes=12) as pool:
        df = pool.starmap(create_mask, paramss)
    
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))