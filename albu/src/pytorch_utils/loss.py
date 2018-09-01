import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F

eps = 1e-3

def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)

def multi_class_dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return multi_class_dice(preds, trues, is_average=is_average)

def dice_loss(preds, trues, weight=None, is_average=True):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


def per_class_dice(preds, trues, weight=None, is_average=True):
    loss = []
    for idx in range(1, preds.shape[1]):
        loss.append(dice_loss(preds[:,idx,...].contiguous(), (trues==idx).float().contiguous(), weight, is_average))
    return loss


def multi_class_dice(preds, trues, weight=None, is_average=True):
    channels = per_class_dice(preds, trues, weight, is_average)
    return sum(channels) / len(channels)


def jaccard_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return jaccard(preds, trues, is_average=is_average)


def jaccard(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return dice_loss(input, target, self.weight, self.size_average)

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return jaccard(input, target, self.weight, self.size_average)

class BCEDiceJaccardLoss(nn.Module):
    def __init__(self, weights, weight=None, size_average=True):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.jacc = JaccardLoss()
        self.dice = DiceLoss()
        self.mapping = {'bce': self.bce,
                        'jacc': self.jacc,
                        'dice': self.dice}
        self.values = {}

    def forward(self, input, target):
        loss = 0
        sigmoid_input = torch.sigmoid(input)
        for k, v in self.weights.items():
            if not v: 
                continue
            val = self.mapping[k](input if k == 'bce' else sigmoid_input, target)
            self.values[k] = val
            if k != 'bce':
                loss += self.weights[k] * (1 - val)
            else:
                loss += self.weights[k] * val
        return loss
