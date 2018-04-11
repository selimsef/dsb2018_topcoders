import keras.backend as K
from keras.losses import categorical_crossentropy


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_dice_coef_ch1(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def double_head_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    return mask_loss + contour_loss


def mask_contour_mask_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    full_mask = dice_coef_loss_bce(y_true[..., 2], y_pred[..., 2])
    return mask_loss + 2 * contour_loss + full_mask


def softmax_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * 0.6 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2


def make_loss(loss_name):
    if loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)

        return loss
    elif loss_name == 'bce':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0, bce=1)

        return loss
    elif loss_name == 'categorical_dice':
        return softmax_dice_loss
    elif loss_name == 'double_head_loss':
        return double_head_loss
    elif loss_name == 'mask_contour_mask_loss':
        return mask_contour_mask_loss
    else:
        ValueError("Unknown loss.")
