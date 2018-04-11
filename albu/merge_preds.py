import os
import tqdm
import numpy as np
import cv2

def merge_files(root):
    res_path = os.path.join('..', '..', 'predictions', os.path.split(root)[-1] + '_test')
    os.makedirs(res_path, exist_ok=True)
    prob_files = {f for f in os.listdir(root) if os.path.splitext(f)[1] in ['.png']}
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    if not unfolded:
        unfolded = prob_files

    for prob_file in tqdm.tqdm(unfolded):
        probs = []
        for fold in range(4):
            prob = os.path.join(root, 'fold{}_'.format(fold) + prob_file)
            prob_arr = cv2.imread(prob, cv2.IMREAD_UNCHANGED)
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)

        res_path_geo = os.path.join(res_path, prob_file)
        cv2.imwrite(res_path_geo, prob_arr)

if __name__ == "__main__":
    val_dir = r'C:\dev\dsbowl\results_test\dpn_softmax_f0'
    merge_files(val_dir)
