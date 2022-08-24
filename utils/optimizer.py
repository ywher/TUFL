#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging

logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                model,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True}]
        self.optim = torch.optim.SGD(
                param_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps) if self.warmup_steps>0 else -1


    def get_lr(self):
        if self.it <= self.warmup_steps and self.warmup_steps>0:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', True):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2 and self.warmup_steps>0:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()



class Optimizer_Adam(object):
    def __init__(self,
                model,
                lr0,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0

        self.optim = torch.optim.Adam(
                model.parameters(),
                lr = lr0,
                betas=(0.9, 0.99),)

        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> Adam warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()