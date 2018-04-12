import sys
from os import path, mkdir, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2
import timeit
from tqdm import tqdm
from skimage import measure
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed

root_data_folder = path.join('..', 'data')

masks_out_folder = path.join(root_data_folder, 'masks_all')
labels_folder = path.join(root_data_folder, 'labels_all')

def create_mask(img_id):
    labels = cv2.imread(path.join(labels_folder, '{0}.tif'.format(img_id)), cv2.IMREAD_UNCHANGED)
    
    tmp = dilation(labels > 0, square(9))    
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(7))
    msk = (255 * tmp).astype('uint8')
    
    props = measure.regionprops(labels)
    msk0 = 255 * (labels > 0)
    msk0 = msk0.astype('uint8')
    
    msk1 = np.zeros_like(labels, dtype='bool')
    
    max_area = np.max([p.area for p in props])
    
    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                if max_area > 4000:
                    sz = 6
                else:
                    sz = 3
            else:
                sz = 3
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 1
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 2
            uniq = np.unique(labels[max(0, y0-sz):min(labels.shape[0], y0+sz+1), max(0, x0-sz):min(labels.shape[1], x0+sz+1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
                msk0[y0, x0] = 0
    
    msk1 = 255 * msk1
    msk1 = msk1.astype('uint8')
    
    msk2 = np.zeros_like(labels, dtype='uint8')
    msk = np.stack((msk0, msk1, msk2))
    msk = np.rollaxis(msk, 0, 3)
    
    cv2.imwrite(path.join(masks_out_folder, '{0}.png'.format(img_id)), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return 0

if __name__ == '__main__':
    t0 = timeit.default_timer()

    fold_nums = [0]

    if not path.isdir(masks_out_folder):
        mkdir(masks_out_folder)
        
    paramss = []
    for f in tqdm(listdir(labels_folder)):
        if '.tif' not in f:
            continue
        paramss.append((f.split('.tif')[0],))
        
    with Pool(processes=12) as pool:
        df = pool.starmap(create_mask, paramss)
    
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))