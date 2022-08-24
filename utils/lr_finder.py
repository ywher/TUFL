from __future__ import print_function, with_statement, division
import copy
import os
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


class LRFinder(object):
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Arguments:
        model (torch.nn.Module): wrapped model.
        optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): wrapped loss function.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        memory_cache (boolean): if this flag is set to True, `state_dict` of model and
            optimizer will be cached in memory. Otherwise, they will be saved to files
            under the `cache_dir`.
        cache_dir (string): path for storing temporary files. If no path is specified,
            system-wide temporary directory is used.
            Notice that this parameter will be ignored if `memory_cache` is True.

    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)

    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai

    """

    def __init__(self, model, optimizer, criterion, device=None, memory_cache=True, cache_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        if device:
            self.device = device
        else:
            self.device = self.model_device

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

    def range_test(
        self,
        train_loader,
        val_loader=None,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.

        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.

        """
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        # Create an iterator to get data batch by batch
        iterator = iter(train_loader)
        for iteration in tqdm(range(num_iter)):
            # Get a new set of inputs and labels
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)

            # Train on batch and retrieve loss
            loss = self._train_batch(inputs, labels)
            if val_loader:
                loss = self._validate(val_loader)

            # Update the learning rate
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def _train_batch(self, inputs, labels):
        # Set model to training mode
        self.model.train()

        # Move data to the correct device
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate(self, dataloader):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Move data to the correct device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        return running_loss / len(dataloader.dataset)

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.

        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.

        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given `cache_dir` is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError('Failed to load state in {}. File does not exist anymore.'.format(fn))
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""
        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])


# !/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import setup_logger
from model import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss, KLDivergenceLoss
from evaluate import evaluate
from optimizer import Optimizer
import torch.nn.functional as F
import torch
import torch.nn as nn
import random
import torch.distributed as dist
from transform_image import affine_transform
import os
import os.path as osp
import logging
import time
import datetime
import argparse
import logging
import os
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from utils import train_curves
from evaluate import iou

respth = './res'
if not osp.exists(respth): os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1)
    parse.add_argument('--number of classes', dest='n_classes', type=int, default=19)
    parse.add_argument('--number of iterations', dest='max_iter', type=int, default=8000)
    parse.add_argument('--number of mesage iterations', dest='msg_iter', type=int, default=50)
    parse.add_argument('--number of train images per gpu', dest='n_img_per_gpu_train', type=int, default=2)
    parse.add_argument('--number of unsupervised images per gpu', dest='n_img_per_gpu_unsup', type=int, default=8)
    parse.add_argument('--number of val images per gpu', dest='n_img_per_gpu_val', type=int, default=1)
    parse.add_argument('--number of workers', dest='n_workers', type=int, default=4)
    parse.add_argument('--root path of the dataset', dest='dataroot', type=str,
                       default='/media/xhq/ubuntu/Dataset/kitti_semantic')  # /media/xhq/ubuntu/Dataset/kitti_semantic #/media/xhq/ubuntu/Dataset/cityscapes_original/gtFine_trainvaltest

    parse.add_argument('--crop size', dest='cropsize', type=int, default=[352, 608])  # [1024,1024]
    parse.add_argument('--dataset', dest='dataset', type=str, default='Kitti_semantics')  # CityScapes#Kitti_semantics
    parse.add_argument('--continue training', dest='continue_training', type=str, \
                       default="/media/xhq/ubuntu/projects/small_thesis/BiSeNet-uda/trained_model/kitti_sup/model_epoch_80000.pth")  # None

    parse.add_argument('--unsupervised', dest='if_unsupervised', type=bool, default=True)
    parse.add_argument('--The threshold on predicted probability on unsupervised data', dest='uda_confidence_thresh',
                       type=float, default=-1)  # 0.5
    parse.add_argument('--The temperature', dest='uda_softmax_temp', type=float, default=-1)  # 0.9
    parse.add_argument('--The coefficient on the UDA loss', dest='unsup_coeff', type=float, default=1.0)

    return parse.parse_args()


def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:33271',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    setup_logger(respth)
    evals = {"train_loss": [], "valid_loss": [], "train_iou": [], "valid_iou": [], "sup_loss": [], "unsup_loss": []}

    ## dataset
    if args.dataset == 'CityScapes':
        # trainset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='train',data_ratio=0.1,resize=(0.5,0.5))
        # valset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='val',data_ratio=1.0,resize=(0.5,0.5))
        # unsupset=CityScapes(args.dataroot, cropsize=args.cropsize, mode='train-u',data_ratio=1.0,resize=(0.5,0.5))

        trainset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='train')
        valset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='val')
        unsupset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='train-u')
    elif args.dataset == 'Kitti_semantics':
        trainset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='train', data_ratio=1.0)
        valset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='val', data_ratio=1.0)
        unsupset = CityScapes(args.dataroot, cropsize=args.cropsize, mode='unsup', data_ratio=1.0)

    sampler_trainset = torch.utils.data.distributed.DistributedSampler(trainset)
    sampler_valset = torch.utils.data.distributed.DistributedSampler(valset)
    sampler_unsupset = torch.utils.data.distributed.DistributedSampler(unsupset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.n_img_per_gpu_train, shuffle=False, num_workers=args.n_workers, pin_memory=True,
        sampler=sampler_trainset, drop_last=True)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.n_img_per_gpu_val, shuffle=False, num_workers=args.n_workers, pin_memory=True,
        sampler=sampler_valset, drop_last=True)

    unsuploader = torch.utils.data.DataLoader(
        unsupset, batch_size=args.n_img_per_gpu_unsup, shuffle=False, num_workers=args.n_workers, pin_memory=True,
        sampler=sampler_unsupset, drop_last=True)

    print("Number of train set: ", len(trainset))
    print("Number of val set: ", len(valset))
    # print("Number of unsup set: ", len(unsupset))

    ## model
    ignore_idx = 255
    net = BiSeNet(n_classes=args.n_classes)
    if args.continue_training is not None:
        net.load_state_dict(torch.load(args.continue_training))
        print("pretrained_model loaded")
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank, ],
                                              output_device=args.local_rank
                                              )
    score_thres = 0.7
    score_thres_val = 0.7
    score_thres_unsup = 0.7
    n_min = args.n_img_per_gpu_train * args.cropsize[0] * args.cropsize[1] // 16
    n_min_val = args.n_img_per_gpu_val * args.cropsize[0] * args.cropsize[1] // 16
    n_min_unsup = args.n_img_per_gpu_unsup * args.cropsize[0] * args.cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_unsup = KLDivergenceLoss(thresh=score_thres_unsup, n_min=n_min_unsup,
                                      uda_softmax_temp=args.uda_softmax_temp,
                                      uda_confidence_thresh=args.uda_confidence_thresh)
    # criteria_val=nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criteria_val = OhemCELoss(thresh=score_thres_val, n_min=n_min_val, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = args.max_iter
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-2)
    lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
    lr_finder.plot()







if __name__ == "__main__":
    train()