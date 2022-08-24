#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging
import numpy as np
import torch.distributed as dist
import math

def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def iou(logits, lb, n_classes = 19,lb_ignore=255):
    logits = logits.cpu().data.numpy()
    preds = np.argmax(logits, axis=1)

    SMOOTH = 1e-16

    lb= lb.data.cpu().numpy()

    ignore_idx = lb_ignore
    keep = np.logical_not(lb == ignore_idx)
    #the merge vaires with predictions and labels changing
    merge = preds[keep] * n_classes + lb[keep]
    hist = np.bincount(merge, minlength=n_classes ** 2)
    hist = hist.reshape((n_classes, n_classes))


    diag=np.diag(hist)
    sum1=np.sum(hist, axis=0)
    sum2=np.sum(hist, axis=1)
    # print("diag ",np.mean(diag))
    # print("sum1 ",np.mean(sum1))
    # print("sum2 ",np.mean(sum2))
    # print("fenmu ",np.mean(sum1 + sum2 - diag))

    ious = (diag+SMOOTH)  / (sum1 + sum2 - diag+SMOOTH )

    ious=ious[ious != 1.0]
    ious = ious[~np.isnan(ious)]
    # print("iou shape ", ious.shape)
    # print("ious ",np.max(ious))

    mean_iou=np.mean(ious) if ious.size!=0 else -1
    # print("mean_iou ", mean_iou)
    #

    # for class_element in range(n_classes):
    #     intersection = (preds & lb & (lb==class_element)).sum((1, 2))
    #     union = ((preds | lb) & (lb==class_element)).sum((1, 2))
    #     iou = (intersection + SMOOTH) / (union + SMOOTH)
    #     mean_iou=np.mean(iou)
    # print("mean_iou ", mean_iou)
    return mean_iou

def weight_computation(freq):
    weight=[1.0 / math.log(1.10 + element) for element in freq]
    return weight