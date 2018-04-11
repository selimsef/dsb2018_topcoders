import os

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np

from .eval import Evaluator


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix=""):
        if self.test:
            path = os.path.join(self.config.dataset_path, name)
        else:
            path = os.path.join(self.config.dataset_path, 'images_all', name)
        rows, cols = cv2.imread(path, 0).shape[:2]
        prediction = prediction[0:rows, 0:cols,...]
        if prediction.shape[2] < 3:
            zeros = np.zeros((rows, cols), dtype=np.float32)
            prediction = np.dstack((prediction[...,0], prediction[...,1], zeros))
        else:
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        if self.test:
            name = os.path.split(name)[-1]
        cv2.imwrite(os.path.join(self.save_dir, prefix + name), (prediction * 255).astype(np.uint8))
