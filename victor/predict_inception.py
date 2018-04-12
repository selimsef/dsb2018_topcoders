from os import path, mkdir, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
import cv2
from models import get_inception_resnet_v2_unet_softmax
from tqdm import tqdm

test_folder = path.join('..', 'data_test')

models_folder = 'nn_models'
test_pred = path.join('..', 'predictions', 'inception_test_pred_4')

all_ids = []
all_images = []
all_masks = []

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127.5
    x -= 1.
    return x

def bgr_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return lab[..., np.newaxis]

if __name__ == '__main__':
    t0 = timeit.default_timer()

    if not path.isdir(test_pred):
        mkdir(test_pred)
        
    print('Loading models')

    models = []
    
    for it in range(4):
        model = get_inception_resnet_v2_unet_softmax((None, None), weights=None)
        model.load_weights(path.join(models_folder, 'inception_resnet_v2_weights_{0}.h5'.format(it)))
        models.append(model)
        
    print('Predicting test')
    for d in tqdm(listdir(test_folder)):
        if not path.isdir(path.join(test_folder, d)):
            continue
        final_mask = None
        for scale in range(3):
            fid = d
            img = cv2.imread(path.join(test_folder, fid, 'images', '{0}.png'.format(fid)), cv2.IMREAD_COLOR)
            if final_mask is None:
                final_mask = np.zeros((img.shape[0], img.shape[1], 3))
            if scale == 1:
                img = cv2.resize(img, None, fx=0.75, fy=0.75)
            elif scale == 2:
                img = cv2.resize(img, None, fx=1.25, fy=1.25)
            elif scale == 3:
                img = cv2.resize(img, None, fx=1.5, fy=1.5)
                
            x0 = 16
            y0 = 16
            x1 = 16
            y1 = 16
            if (img.shape[1] % 32) != 0:
                x0 = int((32 - img.shape[1] % 32) / 2)
                x1 = (32 - img.shape[1] % 32) - x0
                x0 += 16
                x1 += 16
            if (img.shape[0] % 32) != 0:
                y0 = int((32 - img.shape[0] % 32) / 2)
                y1 = (32 - img.shape[0] % 32) - y0
                y0 += 16
                y1 += 16
            img0 = np.pad(img, ((y0,y1), (x0,x1), (0, 0)), 'symmetric')
            
            img0 = np.concatenate([img0, bgr_to_lab(img0)], axis=2)
            
            inp0 = []
            inp1 = []
            for flip in range(2):
                for rot in range(4):
                    if flip > 0:
                        img = img0[::-1, ...]
                    else:
                        img = img0
                    if rot % 2 == 0:
                        inp0.append(np.rot90(img, k=rot))
                    else:
                        inp1.append(np.rot90(img, k=rot))
            
            inp0 = np.asarray(inp0)
            inp0 = preprocess_inputs(inp0)
            inp1 = np.asarray(inp1)
            inp1 = preprocess_inputs(inp1)
            
            mask = np.zeros((img0.shape[0], img0.shape[1], 3))
            
            for model in models:
                pred0 = model.predict(inp0, batch_size=1)
                pred1 = model.predict(inp1, batch_size=1)
                j = -1
                for flip in range(2):
                    for rot in range(4):
                        j += 1
                        if rot % 2 == 0:
                            pr = np.rot90(pred0[int(j / 2)], k=(4-rot))
                        else:
                            pr = np.rot90(pred1[int(j / 2)], k=(4-rot))
                        if flip > 0:
                            pr = pr[::-1, ...]
                        mask += pr
                        
            mask /= (8 * len(models))
            mask = mask[y0:mask.shape[0]-y1, x0:mask.shape[1]-x1, ...]
            if scale > 0:
                mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
            final_mask += mask
        final_mask /= 3
        final_mask = final_mask * 255
        final_mask = final_mask.astype('uint8')
        cv2.imwrite(path.join(test_pred, '{0}.png'.format(fid)), final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))