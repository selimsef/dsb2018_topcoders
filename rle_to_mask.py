import numpy as np
import pandas as pd
import cv2
import os

results_dir = r'D:\dsbowl\test_imgs\labels'

def rle_decode(rle_list, mask_shape, mask_dtype):
    # mask = np.zeros(mask_shape, dtype=mask_dtype)
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for j, rle in enumerate(rle_list):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = (j+1)

    return mask.reshape(mask_shape[::-1]).T


def decode_submission(filename):
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(filename, sep=',')

    for i, test_id in enumerate(pd.unique(df['ImageId'])):
        print('Test ID = ', test_id)
        mask_rles = df.loc[df['ImageId'] == test_id]
        rows, cols = pd.unique(mask_rles['Height'])[0], np.unique(mask_rles['Width'])[0]
        mask = rle_decode(rle_list=mask_rles['EncodedPixels'], mask_shape=(rows, cols), mask_dtype=np.int)
        mask = mask.astype(np.uint16)
        cv2.imwrite(os.path.join(results_dir, test_id + '.tif'), mask)


if __name__ == '__main__':
    filename = 'stage1_solution.csv'
    decode_submission(filename)
