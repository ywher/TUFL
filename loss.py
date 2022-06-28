#!/usr/bin/python
# -*- encoding: utf-8 -*-
from random import random
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import kl_div,softmax,log_softmax, sigmoid
import math
#from datasets.transform_image import *
import sys
MIN_EPSILON=1e-8
LOG_EPSILON=1e-30
class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, weight=None,*args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.softmax = nn.Softmax(dim=1)
        if weight is not None:
            self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none', weight=weight)
        else:
            self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        # print("loss ",loss.size())

        #ohem loss part
        loss =loss.view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

class RegularCrossLoss(nn.Module):
    def __init__(self,  ignore_lb=255,weight=None, *args, **kwargs):
        super(RegularCrossLoss, self).__init__()
        print('in the uda folder')
        if weight is not None:
            self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none',weight=weight)
        else:
            self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        # print("loss ",loss.size())
        assert loss.nelement() > 0, "sup loss is empty　"+ str(np.unique(labels.cpu().numpy()))

        return torch.mean(loss)

class SupCrossEntropyLoss(nn.Module):
    def __init__(self, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0,*args, **kwargs):
        super(SupCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp

    def forward(self, logits_origin, logits_transfer):
        if self.uda_softmax_temp!=-1:
            logits_origin = logits_origin/self.uda_softmax_temp

        assert len(logits_origin) == len(logits_transfer)

        preds_origin=softmax(logits_origin,dim=1)
        preds_transfer=softmax(logits_transfer,dim=1)

        preds_origin_detach = preds_origin.detach() #as the pseudo label
        log_preds_origin_detach = torch.log(preds_origin_detach + LOG_EPSILON).detach()

        log_preds_transfer=torch.log(preds_transfer + LOG_EPSILON)

        loss_kldiv = preds_origin_detach * (log_preds_origin_detach - log_preds_transfer)
        loss_entropy = -torch.mul(preds_origin, torch.log(preds_origin + LOG_EPSILON))

        loss = loss_kldiv + loss_entropy


        n,c,h,w= loss.size()

        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)

            # print("weight_expand shape ",weight_expand.shape)
            loss = loss * weight_expand

        #we must sum class losses and return the mean value instead of calculating the mean right away
        loss = torch.sum(loss, dim=1)
        # print(loss.size())

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_origin,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        loss = torch.clamp(loss, min=MIN_EPSILON)

        return torch.mean(loss)

def save_tensor_image(tensor_img, file_path):
    #tensor_img (1,C,H,W)
    from torchvision import utils as vutils
    assert len(tensor_img.shape)==4 and tensor_img.shape[0] == 1
    tensor_img = tensor_img.clone().detach()
    tensor_img = tensor_img.to(torch.device('cpu'))
    vutils.save_image(tensor_img, file_path)

def get_tsa_threshold(schedule, global_step, num_train_steps, start, end, start_thresh=0.3):
  step_ratio = float(global_step) / float(num_train_steps)
  if schedule == "linear":
    coeff = step_ratio
  elif schedule == "exp":
    scale = 5
    # [exp(-5), exp(0)] = [1e-2, 1]
    coeff = math.exp((step_ratio - 1) * scale)
  elif schedule == "log":
    scale = 5
    # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
    coeff = 1 - math.exp((-step_ratio) * scale)

  #wait until the prediction is getting accurate
  if step_ratio<start_thresh:
      coeff=0
  return coeff * (end - start) + start

class KLDivergenceLoss(nn.Module):
    def __init__(self,uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.3,*args, **kwargs):
        super(KLDivergenceLoss, self).__init__()

        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp=uda_softmax_temp
        self.weight=weight
        self.num_train_steps = num_train_steps
        self.tsa = tsa
        self.start_thresh=start_thresh

    def forward(self, logits_before, logits_after):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp
        else:
            None
        assert len(logits_before) == len(logits_after)

        preds_before=softmax(logits_before,dim=1).detach()
        preds_after=softmax(logits_after,dim=1)

        log_logits_before= torch.log(preds_before + LOG_EPSILON).detach()
        log_logits_after=torch.log(preds_after + LOG_EPSILON)


        # print("preds_before size ",preds_before.size())
        # print("log_logits_after size ", log_logits_after.size())

        loss_kldiv = preds_before*(log_logits_before-log_logits_after)
        n,c,h,w= loss_kldiv.size()

        if self.weight is not None:
            if self.weight.size()[0]==c:
                weight_expand=self.weight.expand(n,h,w,c)
                weight_expand=weight_expand.permute(0,3,1,2)
                # print("weight_expand shape ",weight_expand.shape)
                loss_kldiv = loss_kldiv * weight_expand
            elif self.weight.size() == loss_kldiv.size():
                loss_kldiv = loss_kldiv * self.weight
        # print("loss_kldiv size ",loss_kldiv.size())
        loss = torch.sum(loss_kldiv, dim=1)

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_before,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        loss = torch.clamp(loss, min=MIN_EPSILON)
        return torch.mean(loss)

class KLNewLoss(nn.Module):
    def __init__(self,uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.3,*args, **kwargs):
        super(KLNewLoss, self).__init__()

        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp=uda_softmax_temp
        self.weight=weight
        self.num_train_steps = num_train_steps
        self.tsa = tsa
        self.start_thresh=start_thresh

    def forward(self, logits_before, logits_after):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp
        else:
            None
        assert len(logits_before) == len(logits_after)

        preds_before=softmax(logits_before,dim=1).detach()
        preds_after=softmax(logits_after,dim=1)

        log_logits_before= torch.log(preds_before + LOG_EPSILON).detach()
        log_logits_after=torch.log(preds_after + LOG_EPSILON)

        before_minus_after = preds_before - preds_after
        max_value_predict_before = preds_before.amax(dim=1)
        select_mask = preds_before.eq(max_value_predict_before)
        
        select_difference = (before_minus_after * select_mask)
        select_difference = select_difference.sum(dim=1)

        loss_kldiv = preds_before*(log_logits_before-select_difference.pow(2)*log_logits_after) #0.5, 1, 2
        n,c,h,w= loss_kldiv.size()

        if self.weight is not None:
            assert self.weight.size()[0]==c
            weight_expand=self.weight.expand(n,h,w,c)
            weight_expand=weight_expand.permute(0,3,1,2)

            # print("weight_expand shape ",weight_expand.shape)
            loss_kldiv=loss_kldiv*weight_expand
        # print("loss_kldiv size ",loss_kldiv.size())
        loss = torch.sum(loss_kldiv, dim=1)

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_before,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        loss = torch.clamp(loss, min=MIN_EPSILON)
        return torch.mean(loss)

class MinimumEntropyLoss(nn.Module):
    def __init__(self, weight=None,tsa=None, num_train_steps=-1,uda_confidence_thresh=-1,*args, **kwargs):
        super(MinimumEntropyLoss, self).__init__()
        self.weight=weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()

    def forward(self, logits):
        preds=softmax(logits,dim=1)
        n, c, h, w = preds.size()

        loss=-torch.mul(preds, torch.log(preds + LOG_EPSILON))

        if self.weight is not None:
            assert self.weight.size()[0]==c
            weight_expand=self.weight.expand(n,h,w,c)
            weight_expand=weight_expand.permute(0,3,1,2)
            loss=loss*weight_expand

        loss = torch.sum(loss, dim=1)

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        loss= torch.clamp(loss,min=MIN_EPSILON)
        # loss_coeff = get_tsa_threshold(self.tsa, global_step, self.num_train_steps, 0,1,start_thresh=self.start_thresh)
        # loss=loss*loss_coeff
        # print(loss.size())

        return torch.mean(loss)

class ConfidenceEntropy(nn.Module):
    def __init__(self, weight=None,tsa=None, num_train_steps=-1,uda_confidence_thresh=-1,*args, **kwargs):
        super(ConfidenceEntropy, self).__init__()
        self.weight=weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()

    def forward(self, logits):
        preds=softmax(logits,dim=1)
        n, c, h, w = preds.size()
        #author version
        Q = self.target_distribution(preds)
        loss = -torch.mul(Q, torch.log2(preds+1e-30)) / np.log2(c)
        ###old version
        # log_preds = torch.log(preds + LOG_EPSILON) #log P (N,C,H,W)
        # square_preds = torch.square(preds) #P^2 (N,C,H,W)
        # # print('preds', '\n', preds)
        # f = torch.sum(preds, (3,2)) #f (N,C)
        # f = f.detach() #
        # # print('f size', f.size(), f)
        # # print('square_preds', '\n',square_preds)
        # Q_num = torch.div(square_preds, f.unsqueeze(2).unsqueeze(3)) #(N,C,H,W)
        # # print('Q_num', '\n',Q_num)
        # Q_den = torch.sum(Q_num, 1) #(N,H,W)
        # # print('Q_den size', Q_den.size())
        # # print(Q_den)
        # Q = torch.div(Q_num, Q_den.unsqueeze(1).repeat(1,c,1,1)) #(B,C,H,W)
        # # print('Q', '\n', Q)
        # loss = -torch.mul(Q, log_preds)
        # # print('loss', loss)
        # # loss=-torch.mul(preds, torch.log(preds + LOG_EPSILON))

        if self.weight is not None:
            assert self.weight.size()[0]==c
            weight_expand=self.weight.expand(n,h,w,c)
            weight_expand=weight_expand.permute(0,3,1,2)
            loss=loss*weight_expand

        loss = torch.sum(loss, dim=1)

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds,dim=1)
            # print('predmax','\n',preds_max)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        loss= torch.clamp(loss,min=MIN_EPSILON)
        # loss_coeff = get_tsa_threshold(self.tsa, global_step, self.num_train_steps, 0,1,start_thresh=self.start_thresh)
        # loss=loss*loss_coeff
        # print(loss.size())

        return torch.mean(loss)

    def target_distribution(self, q0):
        n, c, h, w = q0.size()
        q = q0
        q = q.transpose(1, 2).transpose(2, 3).contiguous()  #N,C,H,W ->  1 H W 19
        q = q.view(-1, c)  # n*h*w C
        p = q ** 2 / torch.sum(q, dim=0)  # n*H*W c
        p = p / torch.sum(p, dim=1, keepdim=True)  # n*H*W c
        p = torch.reshape(p, [n, h, w, c])  # n,h,w,c
        p = p.transpose(2, 3).transpose(1, 2).contiguous()
        return p

class MinimumSquareLoss(nn.Module):
    def __init__(self, weight=None, tsa=None, num_train_steps=-1, uda_confidence_thresh=-1, *args, **kwargs):
        super(MinimumSquareLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()

    def forward(self, logits):
        preds = softmax(logits, dim=1)
        n, c, h, w = preds.size()

        loss = torch.pow(preds, 2)

        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)
            loss = loss * weight_expand

        loss = torch.sum(loss, dim=1)

        if self.uda_confidence_thresh != -1:
            preds_max, indices_max = torch.max(preds, dim=1)
            loss_mask = (preds_max > self.uda_confidence_thresh).detach()
            loss = loss[loss_mask]


        return torch.mean(loss) / 2

class UnsupCrossEntropyLoss(nn.Module):
    def __init__(self, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0,*args, **kwargs):
        super(UnsupCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp

    def forward(self, logits_before, logits_after):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)

        preds_before=softmax(logits_before,dim=1)
        preds_after=softmax(logits_after,dim=1)
        #treat logits before as solid reference(pseudo) logits
        preds_before_detach = preds_before.detach()
        log_preds_before = torch.log(preds_before + LOG_EPSILON)
        log_preds_before_detach = log_preds_before.detach()

        # preds_after_detach = preds_after.detach()
        log_preds_after=torch.log(preds_after + LOG_EPSILON)
        # log_preds_after_detach = log_preds_after.detach()

        loss_kldiv = preds_before_detach * (log_preds_before_detach - log_preds_after)
        #loss_entropy = -torch.mul(preds_after, log_preds_after)

        ###old version, calculate entropy in pred_before
        loss_entropy = -torch.mul(preds_before, log_preds_before)

        ###symmetric version, no improvement yet
        # loss_kldiv2 = preds_after_detach * (log_preds_after_detach - log_preds_before)
        # loss_entropy2 = -torch.mul(preds_before, log_preds_before)

        loss= (loss_kldiv + loss_entropy) #+ 0.1 * (loss_kldiv2 + loss_entropy2)

        n,c,h,w = loss.size()

        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)

            # print("weight_expand shape ",weight_expand.shape)
            loss = loss * weight_expand

        #we must sum class losses and return the mean value instead of calculating the mean right away
        loss = torch.sum(loss, dim=1)
        # print(loss.size())

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_before,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]
        
        loss = torch.clamp(loss, min=MIN_EPSILON)

        return torch.mean(loss)

class UnsupHardCrossEntropyLoss(nn.Module):
    def __init__(self, uda_softmax_temp=-1,uda_confidence_thresh=-1,class_ratio=-1,n_class=19,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0,*args, **kwargs):
        super(UnsupHardCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.class_ratio = class_ratio #top 0.1
        self.n_class = n_class
        self.uda_softmax_temp = uda_softmax_temp
        self.class_threshold = np.ones(n_class) * uda_confidence_thresh
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.weight, ignore_index=255, reduction='none')

    def forward(self, logits_before, logits_after):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)

        preds_before=softmax(logits_before,dim=1)
        #treat logits before as solid reference(pseudo) logits
        preds_before_detach = preds_before.detach() #(N,C,H,W)
        pesudo_labels = torch.argmax(preds_before_detach, dim=1) #(N,H,W)

        loss = self.cross_entropy(logits_after, pesudo_labels) #use logits_after rather than after softmax

        #we must sum class losses and return the mean value instead of calculating the mean right away
        if self.uda_confidence_thresh!=-1:
            if self.class_ratio != -1:
                max_items = preds_before_detach.max(dim=1)
                label_pred = max_items[1].data.cpu().numpy()
                logits_pred = max_items[0].data.cpu().numpy()

                logits_cls_dict = {c:[self.class_threshold[c]] for c in range(self.n_class)}
                for cls in range(self.n_class):
                    logits_cls_dict[cls].extend(logits_pred[label_pred==cls])
                    if logits_cls_dict[cls] != None:
                        up_class_thres = np.percentile(logits_cls_dict[cls], 100*(1-self.class_ratio))
                        self.class_threshold[cls] = min(self.class_threshold[cls], up_class_thres)
                self.class_threshold = np.clip(self.class_threshold, 0, 1)

                label_cls_thresh = np.apply_along_axis(lambda x: [self.class_threshold[e] for e in x], 2, label_pred)
                loss_mask = logits_pred > label_cls_thresh
                loss_mask = loss_mask.astype(np.uint8)
                loss_mask = torch.from_numpy(loss_mask)
                loss = loss[loss_mask]
            else:
                preds_max,_=torch.max(preds_before_detach, dim=1)
                loss_mask=(preds_max>self.uda_confidence_thresh).detach()
                loss=loss[loss_mask]
        loss = torch.clamp(loss, min=MIN_EPSILON) 

        return torch.mean(loss)

class UnsupFocalLoss(nn.Module):
    def __init__(self, focal_gamma=1, focal_use_sigmoid=False, focal_type=2, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0, *args, **kwargs):
        super(UnsupFocalLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp
        self.focal_use_sigmoid = focal_use_sigmoid
        self.gamma = focal_gamma
        self.focal_type = focal_type #1 for 1-p_2; 2 for p_hat-p_2
        #print('focal gamma', self.gamma)

    def forward(self, logits_before, logits_after, name=None):
    #def forward(self, logits_before, logits_after):
        #print('use focal')
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)

        if self.focal_use_sigmoid:
            preds_before = sigmoid(logits_before, dim=1)
            preds_after = sigmoid(logits_after, dim=1)
        else:
            preds_before=softmax(logits_before,dim=1)  #softmax is better
            preds_after=softmax(logits_after,dim=1)

        preds_before_detach = preds_before.detach()
        log_preds_before_detach = torch.log(preds_before_detach + LOG_EPSILON).detach()
        log_preds_after=torch.log(preds_after + LOG_EPSILON)

        if self.focal_type == 2:
            before_minus_after = preds_before_detach - preds_after
        elif self.focal_type == 1:
            before_minus_after = torch.ones_like(preds_before_detach) - preds_after
        _, max_index = preds_before_detach.max(dim=1) #preds_before as the pseudo label (N,H,W)
        N,C,_,_ = preds_before_detach.size()
        # select_mask = torch.zeros_like(preds_before_detach)
        # #print(max_index)
        # for i in range(N):
        #     for j in range(C):
        #         select_mask[i][j][max_index[i]==j] = 1
        select_mask = torch.zeros_like(preds_before_detach)
        select_mask.scatter_(1, max_index.unsqueeze(1), 1)
        select_difference = (before_minus_after * select_mask) #(N,C,H,W)
        select_difference = select_difference.sum(dim=1) #(N,H,W)
        select_difference = (select_difference.unsqueeze(1)).repeat(1, C, 1, 1) #(N,1,H,W)->(N,C,H,W)

        loss_kldiv = preds_before_detach * (log_preds_before_detach - (abs(select_difference)).pow(self.gamma) * log_preds_after)
        loss_entropy = -torch.mul(preds_before, torch.log(preds_before + LOG_EPSILON))

        loss= loss_kldiv + loss_entropy

        n,c,h,w= loss.size()

        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)
            loss = loss * weight_expand

        #we must sum class losses and return the mean value instead of calculating the mean right away
        loss = torch.sum(loss, dim=1)

        #use the confidence threshold
        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_before,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]
        # if name != None:
        #     if type(name) == list:
        #         name = name[0]
        #     save_path_mask = '/media/ywh/ubuntu/projects/uda/BiSeNet-uda/outputs/images/tmp_focal/' + str(name) + '.png'
        #     loss_mask = loss_mask.unsqueeze(1)
        #     save_tensor_image(loss_mask, save_path_mask)

        loss = torch.clamp(loss, min=MIN_EPSILON)

        return torch.mean(loss)

class RectUnsupFocalLoss(nn.Module):
    def __init__(self, focal_gamma=1, focal_use_sigmoid=False, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0, *args, **kwargs):
        super(RectUnsupFocalLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp
        self.focal_use_sigmoid = focal_use_sigmoid
        self.gamma = focal_gamma
        #print('focal gamma', self.gamma)

    def forward(self, logits_before, logits_after, name=None):
    #def forward(self, logits_before, logits_after):
        #print('use focal')
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)

        if self.focal_use_sigmoid:
            preds_before = sigmoid(logits_before, dim=1)
            preds_after = sigmoid(logits_after, dim=1)
        else:
            preds_before=softmax(logits_before,dim=1)  #softmax is better
            preds_after=softmax(logits_after,dim=1)

        preds_before_detach = preds_before.detach()
        log_preds_before_detach = torch.log(preds_before_detach + LOG_EPSILON).detach()
        log_preds_after=torch.log(preds_after + LOG_EPSILON)
        log_preds_after_detach = torch.log(preds_after.detach() + LOG_EPSILON).detach()

        before_minus_after = preds_before_detach - preds_after
        max_value_predict_before, max_index = preds_before_detach.max(dim=1)
        N,C,_,_ = preds_before_detach.size()
        select_mask = torch.zeros_like(preds_before_detach)
        #print(max_index)
        for i in range(N):
            for j in range(C):
                select_mask[i][j][max_index[i]==j] = 1
        select_difference = (before_minus_after * select_mask)
        select_difference = select_difference.sum(dim=1)
        select_difference = (select_difference.unsqueeze(1)).repeat(1, C, 1, 1)

        loss_kldiv = preds_before_detach * (log_preds_before_detach - (abs(select_difference)).pow(self.gamma) * log_preds_after)
        loss_entropy = -torch.mul(preds_before, torch.log(preds_before + LOG_EPSILON))

        loss= loss_kldiv + loss_entropy

        n,c,h,w= loss.size()
        #introduce weight in channel/class dimension
        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)
            loss = loss * weight_expand

        #we must sum class losses and return the mean value instead of calculating the mean right away
        loss = torch.sum(loss, dim=1) #sum loss in channel dimension
        # print('loss', loss.shape)
        # print(loss)

        #use variance to update the loss
        # variance = preds_before_detach * (log_preds_before_detach - log_preds_after_detach)
        # variance = torch.sum(variance, dim=1)
        # exp_variance = torch.exp(-variance)
        # loss = loss * exp_variance ###not plus variance

        #use entropy to update the loss
        loss_entropy_detach = -torch.mul(preds_before_detach, torch.log(preds_before_detach + LOG_EPSILON))
        loss_entropy_detach = torch.sum(loss_entropy_detach, dim=1) #sum in channel dimension
        exp_entropy = torch.exp(-loss_entropy_detach)
        #print('max, min, mean', torch.max(loss_entropy_detach),torch.min(loss_entropy_detach),torch.mean(loss_entropy_detach))
        #print('exp max, min, mean', torch.max(exp_entropy),torch.min(exp_entropy),torch.mean(exp_entropy))
        loss = loss * exp_entropy

        #use the confidence threshold to select the loss
        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_before,dim=1)
            loss_mask=(preds_max>float(self.uda_confidence_thresh)).detach()
            loss=loss[loss_mask]
        # if name != None:
        #     if type(name) == list:
        #         name = name[0]
        #     save_path_mask = '/media/ywh/ubuntu/projects/uda/BiSeNet-uda/outputs/tmp_focal/' + str(name) + '.png'
        #     loss_mask = loss_mask.unsqueeze(1)
        #     save_tensor_image(loss_mask, save_path_mask)

        loss = torch.clamp(loss, min=MIN_EPSILON)

        return torch.mean(loss)

class AdaptUnsupFocalLoss(nn.Module):
    def __init__(self, focal_gamma=1, focal_use_sigmoid=False, uda_softmax_temp=-1,uda_confidence_thresh=0.8,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0,n_class = 19, pseudo_pl_alpha=0.2, pseudo_pl_beta=0.9, pseudo_pl_gamma=8.0, beta_decay=0,*args, **kwargs):
        super(AdaptUnsupFocalLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp
        self.focal_use_sigmoid = focal_use_sigmoid
        self.gamma = focal_gamma
        self.n_class = n_class
        self.cls_thresh = (torch.ones(n_class, dtype=torch.float64) * uda_confidence_thresh).cuda() #change np.ones to torch.ones (float64)
        self.pseudo_pl_alpha = pseudo_pl_alpha
        self.pseudo_pl_beta = pseudo_pl_beta #history parameter
        self.pseudo_pl_gamma = pseudo_pl_gamma
        self.beta_decay = beta_decay

        #print('focal gamma', self.gamma)

    def ias_thresh(self, preds_before_detach, n_class, alpha=0.8, w=None, gamma=8.0):
        # if w is None: #self.cls_threshold, global, last iteration
            # w = torch.ones(n_class, dtype=torch.float64).cuda()
        logits_pred, label_pred = preds_before_detach.max(dim=1) #(N,H,W), (N,H,W)
        # max_items = preds_before_detach.max(dim=1)
        # label_pred = max_items[1].data.cpu().numpy()
        # logits_pred = max_items[0].data.cpu().numpy()

        # conf_dict = {c: [self.cls_thresh[c]] for c in range(self.n_class)} #{use past information to initialize}
        # for cls in range(self.n_class):
            # conf_dict[cls].extend(logits_pred[label_pred == cls])
        # threshold 
        # cls_thresh = np.ones(n_class,dtype = np.float32)
        cls_thresh = (torch.ones(n_class, dtype=torch.float64)).cuda()
        for idx_cls in range(n_class):
            conf_tensor_list = logits_pred[label_pred==idx_cls]
            if len(conf_tensor_list) == 0:
                cls_thresh[idx_cls] = self.cls_thresh[idx_cls]
            else:
                conf_tensor_list_extend = torch.ones(conf_tensor_list.size()[0]+1, dtype=conf_tensor_list.dtype).cuda() * self.cls_thresh[idx_cls]
                # conf_tensor_list_extend = conf_tensor_list_extend.cuda()
                conf_tensor_list_extend[:conf_tensor_list.size()[0]] = conf_tensor_list
                conf_tensor_list_extend,_ = torch.sort(conf_tensor_list_extend, descending=True)
                index = int(len(conf_tensor_list_extend) * alpha * self.cls_thresh[idx_cls] ** gamma)
                cls_thresh[idx_cls] = conf_tensor_list_extend[index]
                # arr = np.array(conf_dict[idx_cls])
                # cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * (np.exp(w[idx_cls].cpu()-1)) ** gamma)) ###change for novelty
        # return torch.tensor(cls_thresh, dtype=torch.float64).cuda()
        return cls_thresh

    def forward(self, logits_before, logits_after, name=None, iteration=10000):
    #def forward(self, logits_before, logits_after):
        #print('use focal')
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)

        if self.focal_use_sigmoid:
            preds_before = sigmoid(logits_before, dim=1)
            preds_after = sigmoid(logits_after, dim=1)
        else:
            preds_before=softmax(logits_before,dim=1)  #softmax is better
            preds_after=softmax(logits_after,dim=1)

        preds_before_detach = preds_before.detach()
        log_preds_before_detach = torch.log(preds_before_detach + LOG_EPSILON).detach()
        log_preds_after=torch.log(preds_after + LOG_EPSILON)

        before_minus_after = preds_before_detach - preds_after
        _, max_index = preds_before_detach.max(dim=1)
        N,C,_,_ = preds_before_detach.size()
        select_mask = torch.zeros_like(preds_before_detach)
        select_mask.scatter_(1, max_index.unsqueeze(1), 1)
        # select_mask = torch.zeros_like(preds_before_detach)
        # print(max_index)
        # time1 = time.time()
        # for i in range(N):
            # for j in range(C):
                # select_mask[i][j][max_index[i]==j] = 1
            #select_mask[i] = preds_before_detach[i].ge(max_value_predict_before[0][i]-MIN_EPSILON)
        # print('time1 ', time.time()-time1)
        # time2 = time.time()
        # print('time2 ', time.time()-time2)
        # print('selected mask2', select_mask2)
        # print('selected mask1 == mask2?', select_mask==select_mask2)
        select_difference = (before_minus_after * select_mask)
        select_difference = select_difference.sum(dim=1)
        select_difference = (select_difference.unsqueeze(1)).repeat(1, C, 1, 1)
        loss_kldiv = preds_before_detach * (log_preds_before_detach - (abs(select_difference)).pow(self.gamma) * log_preds_after)
        loss_entropy = -torch.mul(preds_before, torch.log(preds_before + LOG_EPSILON))

        loss= loss_kldiv + loss_entropy

        n,c,h,w= loss.size()

        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)

            loss = loss * weight_expand

        #we must sum class losses and return the mean value instead of calculating the mean right away
        loss = torch.sum(loss, dim=1)

        #update the threshold
        # max_items = preds_before_detach.max(dim=1)
        # label_pred = max_items[1].data.cpu().numpy()
        # logits_pred = max_items[0].data.cpu().numpy()
# 
        # logits_cls_dict = {c: [self.cls_thresh[c]] for c in range(self.n_class)} #{use past information to initialize}
        # for cls in range(self.n_class):
            # logits_cls_dict[cls].extend(logits_pred[label_pred == cls])
            
        #instance adaptive selector
        tmp_cls_thresh = self.ias_thresh(preds_before_detach, alpha=self.pseudo_pl_alpha, n_class=self.n_class, w=self.cls_thresh, gamma=self.pseudo_pl_gamma)
        if self.beta_decay > 0:
            beta = 1 - self.pseudo_pl_beta * math.exp(-iteration*self.beta_decay)
            #1 - 0.9 * exp(-iter*0.01), gradually increase
        else:
            beta = self.pseudo_pl_beta
        self.cls_thresh = beta * self.cls_thresh + (1 - beta) * tmp_cls_thresh
        self.cls_thresh[self.cls_thresh>=1] = 0.999

        # num_pixel = np.zeros(self.n_class)
        # num_selected_pixel = np.zeros(self.n_class)
        #use the confidence threshold
        if self.uda_confidence_thresh!=-1:
            logit_amax, label = preds_before_detach.max(dim=1) #(N,H,W)
            label_cls_thresh = torch.ones_like(logit_amax, dtype=self.cls_thresh.dtype)
            for i in range(self.n_class):
                label_cls_thresh[label==i] = self.cls_thresh[i]
            # print('label_cls_thresh', label_cls_thresh)
            # np_logits = preds_before_detach.data.cpu().numpy()
            # logit = np_logits.transpose(0,2,3,1)#(N,C,H,W)->(N,H,W,C)
            # label = np.argmax(logit, axis=3)
            # logit_amax = np.amax(logit, axis=3)
            # print('max value in logit_max', np.max(logit_amax), np.average(logit_amax))
            # label_cls_thresh = np.apply_along_axis(lambda x: [self.cls_thresh[e] for e in x], 2, label)
            loss_mask = logit_amax > label_cls_thresh
            adapt_ratio = float((torch.sum(loss_mask))) / (loss_mask.size()[0]*loss_mask.size()[1]*loss_mask.size()[2])
            loss_mask_solid = (logit_amax > float(self.uda_confidence_thresh))
            solid_ratio = float(torch.sum(loss_mask_solid)) / (loss_mask_solid.size()[0]*loss_mask_solid.size()[1]*loss_mask_solid.size()[2])
            # loss_mask = loss_mask.astype(np.uint8)
            # loss_mask = torch.from_numpy(loss_mask)
            loss=loss[loss_mask] #(N,H,W)
            
        #     for i in range(self.n_class):
        #         cls_mask = logit_amax[label==i]
        #         num_pixel[i] = len(cls_mask)
        #         num_selected_pixel[i] = len(cls_mask[cls_mask>float(self.uda_confidence_thresh)])
        # each_class_ratio = num_selected_pixel / num_pixel
        # print(each_class_ratio)
        
        # if name != None:
        #     if type(name) == list:
        #         name = name[0]
        #     save_path_mask = '/media/ywh/ubuntu/projects/uda/BiSeNet-uda/outputs/images/adapt_focal/' + str(name) + '.png'
        #     loss_mask = loss_mask.unsqueeze(1)
        #     save_tensor_image(loss_mask, save_path_mask)

        loss = torch.clamp(loss, min=MIN_EPSILON)
        if self.uda_confidence_thresh!=-1:
            return torch.mean(loss), adapt_ratio, solid_ratio
        else:
            return torch.mean(loss)

class SquareCrossEntropyLoss(nn.Module):
    def __init__(self, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0,*args, **kwargs):
        super(SquareCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp

    def forward(self, logits_before, logits_after, global_step):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)


        preds_before=softmax(logits_before,dim=1)
        preds_after=softmax(logits_after,dim=1)

        preds_before_detach = preds_before.detach()

        loss_kldiv = preds_before_detach * (preds_before_detach - preds_after)
        loss_entropy = -torch.mul(preds_before, preds_before)

        loss= loss_kldiv + loss_entropy


        n,c,h,w= loss.size()

        if self.weight is not None:
            assert self.weight.size()[0] == c
            weight_expand = self.weight.expand(n, h, w, c)
            weight_expand = weight_expand.permute(0, 3, 1, 2)

            # print("weight_expand shape ",weight_expand.shape)
            loss = loss * weight_expand

        #we must sum class losses and return the mean value instead of calculating the mean right away
        loss = torch.sum(loss, dim=1)

        # print("max loss ",torch.max(loss))
        # print(loss.size())

        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds_before,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        if self.tsa!=None:
            loss_coeff = get_tsa_threshold(self.tsa, global_step, self.num_train_steps, 0, 1,
                                           start_thresh=self.start_thresh)
            loss = loss * loss_coeff

        loss = torch.clamp(loss, min=MIN_EPSILON)

        return torch.mean(loss)

class AnnealingLoss(nn.Module):
    def __init__(self, tsa='linear', ignore_lb=255, num_train_steps=-1,num_class=19,*args, **kwargs):
        super(AnnealingLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.tsa=tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.tsa_start=1.0/num_class
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')


    def forward(self, logits, labels,global_step):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        # print("loss ",loss.size())
        #annealing loss mask
        eff_train_prob_threshold = get_tsa_threshold( self.tsa, global_step, self.num_train_steps,self.tsa_start, end=1)
        # print("eff_train_prob_threshold ",eff_train_prob_threshold)
        preds=self.softmax(logits)
        preds_max, preds_indices = torch.max(preds, dim=1)

        loss_mask = 1-((preds_indices == labels) * (preds_max > eff_train_prob_threshold))
        loss_mask =loss_mask.detach()
        loss = loss[loss_mask]
        # print("loss ",loss.size())

        if loss.nelement() == 0:
            return 0
        else:
            return torch.mean(loss)

class MutualInformationLoss(nn.Module):
    def __init__(self,coefficient=1.0, half_T_side_dense=10,half_T_side_sparse_min=0,half_T_side_sparse_max=0,*args, **kwargs):
        super(MutualInformationLoss, self).__init__()
        self.coefficient=coefficient
        self.half_T_side_dense=half_T_side_dense
        self.half_T_side_sparse_min=half_T_side_sparse_min
        self.half_T_side_sparse_max=half_T_side_sparse_max
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x1_outs, x2_outs):
        assert (x1_outs.requires_grad)
        assert (x2_outs.requires_grad)
        assert (x1_outs.shape == x2_outs.shape)
        x1_outs=self.softmax(x1_outs)
        x2_outs_inv = self.softmax(x2_outs)

        # bring x2 back into x1's spatial frame
        if (self.half_T_side_sparse_min != 0) or (self.half_T_side_sparse_max != 0):
            x2_outs_inv = random_translation_multiple(x2_outs_inv,
                                                      half_side_min=self.half_T_side_sparse_min,
                                                      half_side_max=self.half_T_side_sparse_max)
        # sum over everything except classes, by convolving x1_outs with x2_outs_inv
        # which is symmetric, so doesn't matter which one is the filter

        bn, k, h, w = x1_outs.shape
        x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

        # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
        # print("x1_outs ", x1_outs.size())
        # print("x2_outs_inv ",x2_outs_inv.size())
        p_i_j = F.conv2d(x1_outs, weight=x2_outs_inv, padding=(self.half_T_side_dense,self.half_T_side_dense))
        # print("first p_i_j.size() ", p_i_j.size())

        # do expectation over each shift location in the T_side_dense *
        # T_side_dense box
        T_side_dense = self.half_T_side_dense * 2 + 1

        # T x T x k x k
        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2,
                                                           keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
        # print("second p_i_j.size() ", p_i_j.size())

        # T x T x k x k
        p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
        p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

        # for log stability; tiny values cancelled out by mult with p_i_j anyway
        p_i_j[(p_i_j < EPS).data] = EPS
        p_i_mat[(p_i_mat < EPS).data] = EPS
        p_j_mat[(p_j_mat < EPS).data] = EPS

        # maximise information
        loss = (-p_i_j * (torch.log(p_i_j) - self.coefficient * torch.log(p_i_mat) -
                          self.coefficient * torch.log(p_j_mat))).sum() / (
                       T_side_dense * T_side_dense)
        # print("p_i_j.size() ",p_i_j.size())
        # print("loss.size() ",loss.size())
        # print("loss ",loss)

        return loss

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class entropyloss(nn.Module):
    def __init__(self):
        super(entropyloss, self).__init__()
    def forward(self, logits, weight):
        """
        logits:     N * C * H * W (before softmax)
        weight:     N * 1 * H * W
        """
        val_num = weight[weight>0].numel()
        logits_log_softmax = torch.log_softmax(logits, dim=1)
        num_classed = logits.size()[1]
        entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
        entropy_reg = torch.sum(entropy) / val_num
        return entropy_reg

class kldloss(nn.Module):
    def __init__(self):
        super(kldloss, self).__init__()
    def forward(self, logits, weight):
        """
        logits:     N * C * H * W 
        weight:     N * 1 * H * W
        """
        val_num = weight[weight>0].numel()
        logits_log_softmax = torch.log_softmax(logits, dim=1)
        num_classes = logits.size()[1]
        kld = - 1/num_classes * weight * logits_log_softmax
        kld_reg = torch.sum(kld) / val_num
        return kld_reg

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=19, ignore_index=255, weight=None):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index=ignore_index
        if weight != None:
            self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none', weight=weight)
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != self.ignore_index).float() #valid position
        labels[labels==self.ignore_index] = self.num_classes #assign ignore label to index class_number
        # zeros = torch.zeros(labels.shape[0],self.num_classes,labels.shape[1],labels.shape[2])
        # label_one_hot = zeros.scatter(0,labels,1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes+1).float().to(self.device) #(N,H,W,C)
        print('label_one_hot',label_one_hot.shape)
        print(label_one_hot)
        print(label_one_hot.permute(0,3,1,2)[:,:-1,:,:])
        label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0) #(N,C,H,W)
        rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1)*mask).sum()/(mask.sum() + 1e-6)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    def forward(self, output, target, pixelWiseWeight):
        #output: (N,C,H,W) before softmax
        #target: (N,H,W)
        loss = self.CE(output, target) #(N,H,W)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss

class KLDivergenceLoss2dPixelWiseWeight(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(KLDivergenceLoss2dPixelWiseWeight, self).__init__()
        self.KL = KLDivergenceLoss(weight=weight)
    def forward(self, pred_mix, mix_pred, pixelWiseWeight):
        #pred_mix: first prediction, then mix, before softmax
        #mix_pred: first_mix, then prediction, before softmax
        loss = self.KL(pred_mix, mix_pred) 
        # print('KL loss size', loss.size())
        loss = torch.mean(loss * pixelWiseWeight)
        return loss

if __name__ == '__main__':
    #if 1:
        # torch.manual_seed(15)
        # criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
        # criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
        # net1 = nn.Sequential(
        #     nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
        # )
        # net1.cuda()
        # net1.train()
        # net2 = nn.Sequential(
        #     nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
        # )
        # net2.cuda()
        # net2.train()
        #
        # with torch.no_grad():
        #     inten = torch.randn(16, 3, 20, 20).cuda()
        #     lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        #     lbs[1, :, :] = 255
        #
        # logits1 = net1(inten)
        # logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
        # logits2 = net2(inten)
        # logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')
        #
        # loss1 = criteria1(logits1, lbs)
        # loss2 = criteria2(logits2, lbs)
        # loss = loss1 + loss2
        # print(loss.detach().cpu())
        # loss.backward()
        # my_softmax=nn.Softmax(dim=1)
        # my_log_softmax = nn.LogSoftmax(dim=1)
        # my_criteria = nn.KLDivLoss(reduction='none')
        # data =torch.tensor([[1.0, 2.0, 3.0],[1.0, 2.0, 3.0],[1.0, 2.0, 3.0],[1.0, 2.0, 3.0]])
        # log_softmax = my_log_softmax(data)
        # print(log_softmax)

        # softmax = my_softmax(data)
        # print(softmax)

        # np_softmax = softmax.data.numpy()
        # log_np_softmax = np.log(np_softmax)
        # print(log_np_softmax)
    import time
    #(2,3,3,3)
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    np.random.seed(1000)
    torch.backends.cudnn.deterministic = True
    predict_before = torch.tensor([[[[0.8,0.1,0.1],[0.8,0.8,0.1],[0.1,0.1,0.8]],[[0.1,0.85,0.2],[0.1,0.1,0.85],[0.85,0.85,0.1]],[[0.1,0.1,0.7],[0.1,0.1,0.1],[0.1,0.1,0.1]]],[[[0.8,0.1,0.1],[0.8,0.8,0.1],[0.1,0.1,0.8]],[[0.1,0.8,0.2],[0.1,0.1,0.8],[0.8,0.8,0.1]],[[0.1,0.1,0.7],[0.1,0.1,0.1],[0.1,0.1,0.1]]]])
    # predict_after = torch.tensor([[[[0.7,0.2,0.1],[0.9,0.7,0.1],[0.2,0.2,0.9]],[[0.2,0.7,0.1],[0.1,0.2,0.9],[0.7,0.7,0.1]],[[0.1,0.1,0.8],[0,0.1,0],[0.1,0.1,0]]],[[[0.7,0.2,0.1],[0.9,0.7,0.1],[0.2,0.2,0.9]],[[0.2,0.7,0.1],[0.1,0.2,0.9],[0.7,0.7,0.1]],[[0.1,0.1,0.8],[0,0.1,0],[0.1,0.1,0]]]])
    predict_before = predict_before.cuda()
    # predict_after = predict_after.cuda()
    # label = torch.tensor([[[0,1,2],[0,255,1],[2,1,0]],[[255,1,2],[0,2,1],[2,1,0]]])
    # tmp = F.adaptive_avg_pool2d(predict_before, 1)
    # print('after adaptive avg',tmp.shape)
    # print(tmp)
    # predict_before = torch.randn(2,19,256,512).cuda()
    # predict_after = torch.randn(2,19,256,512).cuda()
    # print('predict before shape', predict_before.size())
    # print(predict_before)
    # print('predict after shape', predict_after.size())
    #focal = AdaptUnsupFocalLoss(n_class = 3, uda_confidence_thresh=0.51)
    #focal = RectUnsupFocalLoss(uda_confidence_thresh=0.51)
    #focal = UnsupFocalLoss(uda_confidence_thresh=0.8)``
    loss = ConfidenceEntropy(uda_confidence_thresh=0.5, weight=torch.FloatTensor([1]*3).cuda(),n_class=3)
    start_time = time.time()
    for i in range(1):
        a=loss(predict_before)
    print(a)
    end_time = time.time()
    print('total time', end_time-start_time)

