import torch
import numpy as np

# def mcc(y_true, y_pred):
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     tp = K.sum(y_true * y_pred)
#     fn = K.sum(y_true * neg_y_pred)
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     numerator = (tp * tn - fp * fn)
#     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#     return numerator / (denominator + K.epsilon())


def mcc(y_true, y_pred):
    y_pred = y_pred.clone().detach()
    y_true = y_true.clone().detach()
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = torch.sum(y_pos * y_pred_pos)
    tn = torch.sum(y_neg * y_pred_neg)
    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    eps = 1e-10
    return numerator / (denominator + eps)


# def specificity(y_true, y_pred):
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     specificity = tn / (tn + fp + K.epsilon())
#     return specificity
#
#
# def sensitivity(y_true, y_pred):
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     tp = K.sum(y_true * y_pred)
#     fn = K.sum(y_true * neg_y_pred)
#     sensitivity = tp / (tp + fn + K.epsilon())
#     return sensitivity
#
#
# def f1(y_true, y_pred):
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     tp = K.sum(y_true * y_pred)
#     fn = K.sum(y_true * neg_y_pred)
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     sensitivity = tp / (tp + fn + K.epsilon())
#     precision = tp / (tp + fp + K.epsilon())
#     return (2 * ((sensitivity * precision) / (sensitivity + precision + K.epsilon())))
#
