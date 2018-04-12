from os import path, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
import cv2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler #, TensorBoard
from models import get_densenet121_unet_softmax, dice_coef_rounded_ch0, dice_coef_rounded_ch1, schedule_steps, softmax_dice_loss
import keras.backend as K
import pandas as pd
from tqdm import tqdm
from transforms import aug_mega_hardcore
from keras import metrics
from abc import abstractmethod
from keras.preprocessing.image import Iterator
import time
from skimage import measure
from skimage.morphology import square, erosion, dilation, watershed
from skimage.filters import median

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

data_folder = path.join('..', 'data')
masks_folder = path.join(data_folder, 'masks_all')
images_folder = path.join(data_folder, 'images_all')
labels_folder = path.join(data_folder, 'labels_all')
models_folder = 'nn_models'

input_shape = (256, 256)

df = pd.read_csv(path.join(data_folder, 'folds.csv'))

all_ids = []
all_images = []
all_masks = []
all_labels = []
all_good4copy = []

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

def create_mask(labels):
    labels = measure.label(labels, neighbors=8, background=0)
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
    return msk

class BaseMaskDatasetIterator(Iterator):
    def __init__(self,
                 image_ids,
                 random_transformers=None,
                 batch_size=8,
                 shuffle=True,
                 seed=None
                 ):
        self.image_ids = image_ids
        self.random_transformers = random_transformers
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(BaseMaskDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    @abstractmethod
    def transform_mask(self, mask, image):
        raise NotImplementedError

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            _idx = self.image_ids[image_index]
            
            img0 = all_images[_idx].copy()
            msk0 = all_masks[_idx].copy()
            lbl0 = all_labels[_idx].copy()
            good4copy = all_good4copy[_idx]
    
            x0 = random.randint(0, img0.shape[1] - input_shape[1])
            y0 = random.randint(0, img0.shape[0] - input_shape[0])
            img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
            msk = msk0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
            
            if len(good4copy) > 0 and random.random() > 0.75:
                num_copy = random.randrange(1, min(6, len(good4copy)+1))
                lbl_max = lbl0.max()
                for i in range(num_copy):
                    lbl_max += 1
                    l_id = random.choice(good4copy)
                    lbl_msk = all_labels[_idx] == l_id
                    row, col = np.where(lbl_msk)
                    y1, x1 = np.min(np.where(lbl_msk), axis=1)
                    y2, x2 = np.max(np.where(lbl_msk), axis=1)
                    lbl_msk = lbl_msk[y1:y2+1, x1:x2+1]
                    lbl_img = img0[y1:y2+1, x1:x2+1, :]
                    if random.random() > 0.5:
                        lbl_msk = lbl_msk[:, ::-1, ...]
                        lbl_img = lbl_img[:, ::-1, ...]
                    rot = random.randrange(4)
                    if rot > 0:
                        lbl_msk = np.rot90(lbl_msk, k=rot)
                        lbl_img = np.rot90(lbl_img, k=rot)
                    x1 = random.randint(max(0, x0 - lbl_msk.shape[1] // 2), min(img0.shape[1] - lbl_msk.shape[1], x0 + input_shape[1] - lbl_msk.shape[1] // 2))
                    y1 = random.randint(max(0, y0 - lbl_msk.shape[0] // 2), min(img0.shape[0] - lbl_msk.shape[0], y0 + input_shape[0] - lbl_msk.shape[0] // 2))
                    tmp = erosion(lbl_msk, square(5))
                    lbl_msk_dif = lbl_msk ^ tmp
                    tmp = dilation(lbl_msk, square(5))
                    lbl_msk_dif = lbl_msk_dif | (tmp ^ lbl_msk)
                    lbl0[y1:y1+lbl_msk.shape[0], x1:x1+lbl_msk.shape[1]][lbl_msk] = lbl_max
                    img0[y1:y1+lbl_msk.shape[0], x1:x1+lbl_msk.shape[1]][lbl_msk] = lbl_img[lbl_msk]
                    full_diff_mask = np.zeros_like(img0[..., 0], dtype='bool')
                    full_diff_mask[y1:y1+lbl_msk.shape[0], x1:x1+lbl_msk.shape[1]] = lbl_msk_dif
                    img0[..., 0][full_diff_mask] = median(img0[..., 0], mask=full_diff_mask)[full_diff_mask]
                    img0[..., 1][full_diff_mask] = median(img0[..., 1], mask=full_diff_mask)[full_diff_mask]
                    img0[..., 2][full_diff_mask] = median(img0[..., 2], mask=full_diff_mask)[full_diff_mask]
                img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                lbl = lbl0[y0:y0+input_shape[0], x0:x0+input_shape[1]]
                msk = create_mask(lbl)
                
            if 'ic100_' in all_ids[_idx] or 'gnf_' in all_ids[_idx]:
                data = self.random_transformers[1](image=img[..., ::-1], mask=msk)
            else:
                data = self.random_transformers[0](image=img[..., ::-1], mask=msk)
                
            img = data['image'][..., ::-1]
            msk = data['mask']
            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk

            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            batch_x.append(img)
            batch_y.append(otp)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        batch_x = preprocess_inputs(batch_x)
        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    def transform_batch_x(self, batch_x):
        return batch_x


    def next(self):

        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)    
    
def val_data_generator(val_idx, batch_size, validation_steps):
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for i in val_idx:
            img = all_images[i]
            msk = all_masks[i].copy()
            
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
                
            img = np.pad(img, ((y0,y1), (x0,x1), (0, 0)), 'symmetric')
            msk = np.pad(msk, ((y0,y1), (x0,x1), (0, 0)), 'symmetric')
            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk

            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            for j in range(batch_size):
                inputs.append(img)
                outputs.append(otp)
            if len(inputs) == batch_size:
                step_id += 1
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')
                inputs = preprocess_inputs(inputs)
                yield inputs, outputs
                inputs = []
                outputs = []
                if step_id == validation_steps:
                    break
                

def is_grayscale(image):
    return np.allclose(image[..., 0], image[..., 1], atol=0.001) and np.allclose(image[..., 1], image[..., 2], atol=0.001)
                
if __name__ == '__main__':
    t0 = timeit.default_timer()

    fold_nums = [0, 1, 2, 3]

    if not path.isdir(models_folder):
        mkdir(models_folder)

    all_ids = df['img_id'].values
    all_sources = df['source'].values
    
    for i in tqdm(range(len(all_ids))):
        img_id = all_ids[i]
        msk = cv2.imread(path.join(masks_folder, '{0}.png'.format(img_id)), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(path.join(images_folder, '{0}.png'.format(img_id)), cv2.IMREAD_COLOR)
        lbl = cv2.imread(path.join(labels_folder, '{0}.tif'.format(img_id)), cv2.IMREAD_UNCHANGED)
        if img.shape[0] < 256 or img.shape[1] < 256:
            y_pad = 0
            x_pad = 0
            if img.shape[1] < 256:
                x_pad = 256 - img.shape[1]
            if img.shape[0] < 256:
                y_pad = 256 - img.shape[0]
            img = np.pad(img, ((0, y_pad), (0, x_pad), (0, 0)), 'constant')
            msk = np.pad(msk, ((0, y_pad), (0, x_pad), (0, 0)), 'constant')
            lbl = np.pad(lbl, ((0, y_pad), (0, x_pad)), 'constant')
        all_images.append(img)
        all_masks.append(msk)
        all_labels.append(lbl)
        
        tmp = np.zeros_like(msk[..., 0], dtype='uint8')
        tmp[1:-1, 1:-1] = msk[1:-1, 1:-1, 0]
        good4copy = list(set(np.unique(lbl[lbl > 0])).symmetric_difference(np.unique(lbl[(lbl > 0) & (tmp == 0)])))
        all_good4copy.append(good4copy)
            
    batch_size = 16
    val_batch = 1

    polosa_id = '193ffaa5272d5c421ae02130a64d98ad120ec70e4ed97a72cdcd4801ce93b066'
    
    for it in range(4):
        if it not in fold_nums:
            continue
        
        
        train_idx0 = df[(df['fold'] != it) | (df['img_id'] == polosa_id)].index.values
        train_groups = df[(df['fold'] != it) | (df['img_id'] == polosa_id)]['cluster'].values
        train_ids = df[(df['fold'] != it) | (df['img_id'] == polosa_id)]['img_id'].values
        
        train_idx = []
        for i in range(len(train_idx0)):
            rep = 1
            if train_groups[i] in ['b', 'd', 'e', 'n']:
                rep = 3
            elif train_groups[i] in ['c', 'g', 'k', 'l']:
                rep = 2
            if train_ids[i] == polosa_id:
                rep = 5
            train_idx.extend([train_idx0[i]] * rep)
        train_idx = np.asarray(train_idx)
        
        val_idx0 = df[(df['fold'] == it)].index.values
        val_groups = df[(df['fold'] == it)]['cluster'].values
        val_idx = []
        for i in range(len(val_idx0)):
            rep = 1
            if val_groups[i] in ['b', 'd', 'e', 'n']:
                rep = 3
            elif val_groups[i] in ['c', 'g', 'k', 'l']:
                rep = 2
            val_idx.extend([val_idx0[i]] * rep)
        val_idx = np.asarray(val_idx) 
        
        validation_steps = len(val_idx)
        steps_per_epoch = 5 * int(len(train_idx) / batch_size)

        print('Training fold', it)
        print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

        data_gen = BaseMaskDatasetIterator(train_idx,
                     random_transformers=[aug_mega_hardcore((-0.25, 0.6)), aug_mega_hardcore((-0.6, 0.25))],
                     batch_size=batch_size,
                     shuffle=True,
                     seed=1
                     )
        
        np.random.seed(it+111)
        random.seed(it+111)
        tf.set_random_seed(it+111)
        
#        tbCallback = TensorBoard(log_dir="tb_logs/densenet_softmax_{0}".format(it), histogram_freq=0, write_graph=True, write_images=False)
        
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))
        
        model = get_densenet121_unet_softmax((None, None), weights='imagenet')
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=3e-4, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])
        model.fit_generator(generator=data_gen,
                                epochs=6, steps_per_epoch=steps_per_epoch, verbose=2,
                                validation_data=val_data_generator(val_idx, val_batch, validation_steps),
                                validation_steps=validation_steps,
                                callbacks=[lrSchedule],
                                max_queue_size=5,
                                workers=6)

        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 90)]))
        for l in model.layers:
            l.trainable = True
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=5e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])
        model_checkpoint = ModelCheckpoint(path.join(models_folder, 'densenet_weights_{0}.h5'.format(it)), monitor='val_loss', 
                                            save_best_only=True, save_weights_only=True, mode='min')
        model.fit_generator(generator=data_gen,
                                epochs=90, steps_per_epoch=steps_per_epoch, verbose=2,
                                validation_data=val_data_generator(val_idx, val_batch, validation_steps),
                                validation_steps=validation_steps,
                                callbacks=[lrSchedule, model_checkpoint], #, tbCallback
                                max_queue_size=5,
                                workers=6)

        del model
        del model_checkpoint
        K.clear_session()
        
        np.random.seed(it+222)
        random.seed(it+222)
        tf.set_random_seed(it+222)
        
        model = get_densenet121_unet_softmax((None, None), weights=None)
        model.load_weights(path.join(models_folder, 'densenet_weights_{0}.h5'.format(it)))
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-6, 92), (3e-5, 100), (2e-5, 120), (1e-5, 130)]))
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=1e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])
        model_checkpoint2 = ModelCheckpoint(path.join(models_folder, 'densenet_weights_{0}.h5'.format(it)), monitor='val_loss', 
                                            save_best_only=True, save_weights_only=True, mode='min')
        model.fit_generator(generator=data_gen,
                                epochs=130, steps_per_epoch=steps_per_epoch, verbose=2,
                                validation_data=val_data_generator(val_idx, val_batch, validation_steps),
                                validation_steps=validation_steps,
                                callbacks=[lrSchedule, model_checkpoint2], #, tbCallback
                                max_queue_size=5,
                                workers=6,
                                initial_epoch=90)
        
        del model
        del model_checkpoint2
        K.clear_session()
        
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))