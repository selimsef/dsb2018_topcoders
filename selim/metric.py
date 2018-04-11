from multiprocessing.pool import Pool

import numpy as np
from skimage import measure


def calculate_cell_score_kaggle(y_true, y_pred, num_threads=32):
    yps = []
    for m in range(len(y_true)):
        yps.append((y_true[m].copy(), y_pred[m].copy()))
    pool = Pool(num_threads)
    results = pool.map(score_kaggle, yps)
    return np.mean(results)


def calculate_cell_score_selim(y_true, y_pred, num_threads=32, ids=None):
    yps = []
    for m in range(len(y_true)):
        yps.append((y_true[m].copy(), y_pred[m].copy()))
    pool = Pool(num_threads)
    results = pool.map(calculate_jaccard, yps)
    if ids:
        import pandas as pd
        s_iou = np.argsort(results)
        d = []
        for i in range(len(s_iou)):
            id = ids[s_iou[i]]
            res = results[s_iou[i]]
            d.append([id, res])
            pd.DataFrame(d, columns=["ID", "METRIC_SCORE"]).to_csv("gt_vs_oof.csv", index=False)

    return np.array(results).mean()


def get_cells(mask):
    return measure.label(mask, return_num=True)

def score_kaggle(yp):
    y, p = yp
    return calc_score(np.expand_dims(y, 0), np.expand_dims(p, 0))



def calc_score(labels, y_pred):
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    #    print("Number of true objects:", true_objects)
    #    print("Number of predicted objects:", pred_objects)
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    #    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        #        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    #    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def calculate_jaccard(yps):
    y, p = yps
    jaccards = []
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for iou_threshold in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0
        processed_gt = set()
        matched = set()
        size = p.shape[0], p.shape[1]
        mask_img = np.reshape(p, size)

        gt_mask_img = np.reshape(y, size)
        predicted_labels, predicted_count = get_cells(mask_img)
        gt_labels, gt_count = get_cells(gt_mask_img)

        gt_cells = [rp.coords for rp in measure.regionprops(gt_labels)]
        pred_cells = [rp.coords for rp in measure.regionprops(predicted_labels)]
        gt_cells = [to_point_set(b) for b in gt_cells]
        pred_cells = [to_point_set(b) for b in pred_cells]
        for j in range(predicted_count):
            match_found = False
            for i in range(gt_count):
                pred_ind = j + 1
                gt_ind = i + 1
                if match_found:
                    break
                if gt_ind in processed_gt:
                    continue
                pred_cell = pred_cells[j]
                gt_cell = gt_cells[i]
                intersection = len(pred_cell.intersection(gt_cell))
                union = len(pred_cell) + len(gt_cell) - intersection
                iou = intersection / union
                if iou > iou_threshold:
                    processed_gt.add(gt_ind)
                    matched.add(pred_ind)
                    match_found = True
                    tp += 1
            if not match_found:
                fp += 1
        fn += gt_count - len(processed_gt)
        jaccards.append(tp / (tp + fp + fn))
    return np.mean(jaccards)


def to_point_set(cell):
    return set([(row[0], row[1]) for row in cell])
