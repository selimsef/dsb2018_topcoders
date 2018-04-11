import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""

import cv2
from keras.preprocessing.image import load_img, img_to_array
from skimage.color import label2rgb
from tqdm import tqdm

from postprocessing import label_mask

import numpy as np
import pandas as pd


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(lab_img):
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def main():
    TEST_PATH = '/home/selim/kaggle/datasets/bowl/stage1_test/'
    test_ids = next(os.walk(TEST_PATH))[1]
    all_masks = []
    print('Getting and resizing test images ... ')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        pred_mask = img_to_array(load_img('predict/' + id_ + '.png'))[..., ::-1] / 255.
        mask_img = label_mask(pred_mask[..., 0], pred_mask[..., 1], 0.35, 0.35)
        all_masks.append(mask_img)
        clr_labels = label2rgb(mask_img, bg_label=0)
        clr_labels *= 255
        clr_labels = clr_labels.astype('uint8')
        cv2.imwrite('color_predict/' + id_ + '.png', clr_labels)

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(all_masks[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('resnet101.csv', index=False)


if __name__ == '__main__':


    def run_length_decode(rle, H, W, fill_value=255):
        mask = np.zeros((H * W), np.uint8)
        rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
        for r in rle:
            start = r[0] - 1
            end = start + r[1]
            mask[start: end] = fill_value
        mask = mask.reshape(W, H).T  # H, W need to swap as transposing.
        return mask


    img = cv2.imread("predict/3c4c675825f7509877bc10497f498c9a2e3433bf922bd870914a2eb21a54fd26.png")
