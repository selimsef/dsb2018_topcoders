import random
import numpy as np
import matplotlib.pyplot as plt

class ImageCropper:
    def __init__(self, img_rows, img_cols, target_rows, target_cols, pad):
        self.image_rows = img_rows
        self.image_cols = img_cols
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.pad = pad
        self.use_crop = (img_rows != target_rows) or (img_cols != target_cols)
        self.starts_y = self.sequential_starts(axis=0) if self.use_crop else [0]
        self.starts_x = self.sequential_starts(axis=1) if self.use_crop else [0]
        self.positions = [(x, y) for x in self.starts_x for y in self.starts_y]
        # self.lock = threading.Lock()

    def random_crop_coords(self):
        x = random.randint(0, self.image_cols - self.target_cols)
        y = random.randint(0, self.image_rows - self.target_rows)
        return x, y

    def crop_image(self, image, x, y):
        return image[y: y+self.target_rows, x: x+self.target_cols,...] if self.use_crop else image

    def sequential_crops(self, img):
        for startx in self.starts_x:
            for starty in self.starts_y:
                yield self.crop_image(img, startx, starty)

    def sequential_starts(self, axis=0):
        big_segment = self.image_cols if axis else self.image_rows
        small_segment = self.target_cols if axis else self.target_rows
        if big_segment == small_segment:
            return [0]
        steps = np.ceil((big_segment - self.pad) / (small_segment - self.pad)) # how many small segments in big segment
        if steps == 1:
            return [0]
        new_pad = int(np.floor((small_segment * steps - big_segment) / (steps - 1))) # recalculate pad
        starts = [i for i in range(0, big_segment - small_segment, small_segment - new_pad)]
        starts.append(big_segment - small_segment)
        return starts

import random
from skimage.morphology import square, erosion, dilation, watershed
from skimage.filters import median
from skimage import measure

class DVCropper:
    def __init__(self, masks, labels, target_rows, target_cols):
        self.input_shape = (target_rows, target_cols)

        self.all_good4copy = []
        self.all_labels = labels
        for msk, lbl in zip(masks, labels):
            tmp = np.zeros_like(msk[..., 0], dtype='uint8')
            tmp[1:-1, 1:-1] = msk[1:-1, 1:-1, 2]
            good4copy = list(set(np.unique(lbl[lbl > 0])).symmetric_difference(np.unique(lbl[(lbl > 0) & (tmp == 0)])))
            self.all_good4copy.append(good4copy)

    def create_mask(self, labels):
        labels = measure.label(labels, neighbors=8, background=0)
        tmp = dilation(labels > 0, square(9))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = dilation(tmp, square(7))

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
                uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                                 max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
                    msk0[y0, x0] = 0

        msk1 = 255 * msk1
        msk1 = msk1.astype('uint8')

        msk2 = np.zeros_like(labels, dtype='uint8')
        msk = np.stack((msk2, msk1, msk0))
        msk = np.rollaxis(msk, 0, 3)
        return msk

    def strange_method(self, _idx, img0, msk0, lbl0, x0, y0):
        input_shape = self.input_shape
        good4copy = self.all_good4copy[_idx]

        img = img0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
        msk = msk0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]

        if len(good4copy) > 0 and random.random() > 0.75:
            num_copy = random.randrange(1, min(6, len(good4copy) + 1))
            lbl_max = lbl0.max()
            for i in range(num_copy):
                lbl_max += 1
                l_id = random.choice(good4copy)
                lbl_msk = self.all_labels[_idx] == l_id
                row, col = np.where(lbl_msk)
                y1, x1 = np.min(np.where(lbl_msk), axis=1)
                y2, x2 = np.max(np.where(lbl_msk), axis=1)
                lbl_msk = lbl_msk[y1:y2 + 1, x1:x2 + 1]
                lbl_img = img0[y1:y2 + 1, x1:x2 + 1, :]
                if random.random() > 0.5:
                    lbl_msk = lbl_msk[:, ::-1, ...]
                    lbl_img = lbl_img[:, ::-1, ...]
                rot = random.randrange(4)
                if rot > 0:
                    lbl_msk = np.rot90(lbl_msk, k=rot)
                    lbl_img = np.rot90(lbl_img, k=rot)
                x1 = random.randint(max(0, x0 - lbl_msk.shape[1] // 2),
                                    min(img0.shape[1] - lbl_msk.shape[1], x0 + input_shape[1] - lbl_msk.shape[1] // 2))
                y1 = random.randint(max(0, y0 - lbl_msk.shape[0] // 2),
                                    min(img0.shape[0] - lbl_msk.shape[0], y0 + input_shape[0] - lbl_msk.shape[0] // 2))
                tmp = erosion(lbl_msk, square(5))
                lbl_msk_dif = lbl_msk ^ tmp
                tmp = dilation(lbl_msk, square(5))
                lbl_msk_dif = lbl_msk_dif | (tmp ^ lbl_msk)
                lbl0[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]][lbl_msk] = lbl_max
                img0[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]][lbl_msk] = lbl_img[lbl_msk]
                full_diff_mask = np.zeros_like(img0[..., 0], dtype='bool')
                full_diff_mask[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]] = lbl_msk_dif
                img0[..., 0][full_diff_mask] = median(img0[..., 0], mask=full_diff_mask)[full_diff_mask]
                img0[..., 1][full_diff_mask] = median(img0[..., 1], mask=full_diff_mask)[full_diff_mask]
                img0[..., 2][full_diff_mask] = median(img0[..., 2], mask=full_diff_mask)[full_diff_mask]
            img = img0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
            lbl = lbl0[y0:y0 + input_shape[0], x0:x0 + input_shape[1]]
            msk = self.create_mask(lbl)
        return img, msk

#dbg functions
def starts_to_mpl(starts, t):
    ends = np.array(starts) + t
    data = []
    prev_e = None
    for idx, (s, e) in enumerate(zip(starts, ends)):
        # if prev_e is not None:
        #     data.append((prev_e, s))
        #     data.append((idx-1, idx-1))
        #     data.append('b')
        #     data.append((prev_e, s))
        #     data.append((idx, idx))
        #     data.append('b')
        data.append((s, e))
        data.append((idx, idx))
        data.append('r')

        prev_e = e
        if idx > 0:
            data.append((s, s))
            data.append((idx-1, idx))
            data.append('g--')
        if idx < len(starts) - 1:
            data.append((e, e))
            data.append((idx, idx+1))
            data.append('g--')

    return data

def calc_starts_and_visualize(c, tr, tc):
    starts_rows = c.sequential_starts(axis=0)
    data_rows = starts_to_mpl(starts_rows, tr)
    starts_cols = c.sequential_starts(axis=1)
    data_cols = starts_to_mpl(starts_cols, tc)

    f, axarr = plt.subplots(1, 2, sharey=True)
    axarr[0].plot(*data_rows)
    axarr[0].set_title('rows')
    axarr[1].plot(*data_cols)
    axarr[1].set_title('cols')
    plt.show()


if __name__ == '__main__':
    # opts = 2072, 2072, 1024, 1024, 0
    # c = ImageCropper(*opts)
    # calc_starts_and_visualize(c, opts[2], opts[3])
    import cv2, os
    from scipy.misc import imread
    root = r'/home/albu/dev/bowl/train_imgs'
    root_masks = os.path.join(root, 'masks_all6')
    root_labels = os.path.join(root, 'labels_all6')
    root_images = os.path.join(root, 'images_all6')
    masks, labels, images = [], [], []
    for fn in os.listdir(root_masks):
        masks.append(imread(os.path.join(root_masks, os.path.splitext(fn)[0] + '.png'), mode='RGB'))
        images.append(imread(os.path.join(root_images, os.path.splitext(fn)[0] + '.png'), mode='RGB'))
        labels.append(cv2.imread(os.path.join(root_labels, os.path.splitext(fn)[0] + '.tif'), cv2.IMREAD_UNCHANGED))
    c = DVCropper(masks, labels, 256, 256)
    for _idx in range(100):
        img0 = images[_idx]
        msk0 = masks[_idx]
        lbl0 = labels[_idx]
        x0, y0 = 10, 10
        img, msk = c.strange_method(_idx, np.copy(img0), np.copy(msk0), lbl0, x0, y0)
        cv2.imshow('img0', img0[x0:x0+256, y0:y0+256, :])
        cv2.imshow('msk0', msk0[x0:x0+256, y0:y0+256,:])
        cv2.imshow('img', img)
        cv2.imshow('msk', msk)
        cv2.waitKey()
