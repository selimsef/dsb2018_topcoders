import os
from params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

from models.model_factory import make_model


from os import path, mkdir, listdir
import numpy as np

np.random.seed(1)
import random

random.seed(1)
import tensorflow as tf

tf.set_random_seed(1)
import timeit
import cv2
import pandas as pd
from tqdm import tqdm

images_folder = args.images_dir
train_pred = os.path.join(args.out_root_dir, args.out_masks_folder)

df = pd.read_csv(args.folds_csv)

all_ids = []
all_images = []
all_masks = []

weights = [os.path.join(args.models_dir, m) for m in args.models]

OUT_CHANNELS = args.out_channels

def preprocess_inputs(x):
    return preprocess_input(x, mode=args.preprocessing_function)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    fold_nums = [0,1,2,3]  # , 2, 3

    if not path.isdir(train_pred):
        mkdir(train_pred)

    df = df.reset_index().copy()
    all_ids = df['img_id'].values
    all_sources = df['source'].values

    for i in tqdm(range(len(all_ids))):
        img_id = all_ids[i]
        img = img_to_array(load_img(os.path.join(args.images_dir, img_id + ".png")))

        all_images.append(img)

    model = make_model(args.network, (None, None, 3))

    for it in range(4):
        if it not in fold_nums:
            continue

        val_idx = df[(df['fold'] == it)].index.values  # (df['hold_out'] == 0) &

        model.load_weights(weights[it])
        print('Predicting fold', it)
        #        if it < 2:
        #            continue
        for i in tqdm(val_idx):
            final_mask = None
            for scale in range(1):
                fid = all_ids[i]
                img = all_images[i]
                if final_mask is None:
                    final_mask = np.zeros((img.shape[0], img.shape[1], OUT_CHANNELS))
                if scale == 1:
                    img = cv2.resize(img, None, fx=0.75, fy=0.75)
                elif scale == 2:
                    img = cv2.resize(img, None, fx=1.25, fy=1.25)
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
                img0 = np.pad(img, ((y0, y1), (x0, x1), (0, 0)), 'symmetric')


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

                mask = np.zeros((img0.shape[0], img0.shape[1], OUT_CHANNELS))

                pred0 = model.predict(inp0, batch_size=1)
                pred1 = model.predict(inp1, batch_size=1)
                j = -1
                for flip in range(2):
                    for rot in range(4):
                        j += 1
                        if rot % 2 == 0:
                            pr = np.rot90(pred0[int(j / 2)], k=(4 - rot))
                        else:
                            pr = np.rot90(pred1[int(j / 2)], k=(4 - rot))
                        if flip > 0:
                            pr = pr[::-1, ...]
                        mask += pr  # [..., :2]

                mask /= 8
                mask = mask[y0:mask.shape[0] - y1, x0:mask.shape[1] - x1, ...]
                if scale > 0:
                    mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
                final_mask += mask
            final_mask /= 1
            final_mask = final_mask * 255
            final_mask = final_mask.astype('uint8')
            if OUT_CHANNELS == 2:
                final_mask = np.concatenate([final_mask, np.zeros_like(final_mask)[..., 0:1]], axis=-1)
            cv2.imwrite(path.join(train_pred, '{0}.png'.format(fid)), final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])