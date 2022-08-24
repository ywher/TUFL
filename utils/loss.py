#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax, sigmoid
import math
MIN_EPSILON=1e-8
LOG_EPSILON=1e-30

#supervised ohem
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

#supervised cross entropy loss
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

#shannon entropy
class MinimumEntropyLoss(nn.Module):
    def __init__(self, weight=None,tsa=None, num_train_steps=-1,uda_confidence_thresh=-1,ohem=False, ohem_ratio=0.25, *args, **kwargs):
        super(MinimumEntropyLoss, self).__init__()
        self.weight=weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.ohem = ohem #default false
        self.ohem_ratio = ohem_ratio #default 0.25

    def forward(self, logits):
        preds=softmax(logits,dim=1)
        n, c, h, w = preds.size()

        loss=-torch.mul(preds, torch.log(preds + LOG_EPSILON))

        if self.weight is not None:
            assert self.weight.size()[0]==c
            weight_expand=self.weight.expand(n,h,w,c)
            weight_expand=weight_expand.permute(0,3,1,2)
            loss=loss*weight_expand

        loss = torch.sum(loss, dim=1) #[1, 640, 1280], 640*1280=819,200
        #uda confidence threshold
        if self.uda_confidence_thresh!=-1:
            preds_max,indices_max=torch.max(preds,dim=1)
            loss_mask=(preds_max>self.uda_confidence_thresh).detach() #673701 
            loss=loss[loss_mask] #[673701], [num of ones in mask]
        loss= torch.clamp(loss,min=MIN_EPSILON)
        if self.ohem:
            loss =loss.view(-1) #[680177]
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:int(loss.size()[0] * self.ohem_ratio)]
        return torch.mean(loss)

#confidence entropy
class ConfidenceEntropy(nn.Module):
    def __init__(self, weight=None,tsa=None, num_train_steps=-1,uda_confidence_thresh=-1,ohem=False, ohem_ratio=0.25,*args, **kwargs):
        super(ConfidenceEntropy, self).__init__()
        self.weight=weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.ohem = ohem #default false
        self.ohem_ratio = ohem_ratio #default 0.25

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
            loss_mask=(preds_max>self.uda_confidence_thresh).detach()
            loss=loss[loss_mask]

        loss= torch.clamp(loss,min=MIN_EPSILON)
        if self.ohem:
            loss =loss.view(-1) #[680177]
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:int(loss.size()[0] * self.ohem_ratio)]

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

#maximum square loss
class MinimumSquareLoss(nn.Module):
    def __init__(self, weight=None, tsa=None, num_train_steps=-1, uda_confidence_thresh=-1, ohem=False, ohem_ratio=0.25, *args, **kwargs):
        super(MinimumSquareLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.ohem = ohem #default false
        self.ohem_ratio = ohem_ratio #default 0.25

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
            
        loss= torch.clamp(loss,min=MIN_EPSILON)
        if self.ohem:
            loss =loss.view(-1) #[680177]
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:int(loss.size()[0] * self.ohem_ratio)]

        return torch.mean(loss) / 2

#neutral cross entropy loss
class UnsupCrossEntropyLoss(nn.Module):
    def __init__(self, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0, ohem=False, ohem_ratio=0.25,*args, **kwargs):
        super(UnsupCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp
        self.ohem = ohem #default false
        self.ohem_ratio = ohem_ratio #default 0.25

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

        log_preds_after=torch.log(preds_after + LOG_EPSILON)

        loss_kldiv = preds_before_detach * (log_preds_before_detach - log_preds_after)

        loss_entropy = -torch.mul(preds_before, log_preds_before)

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

        if self.ohem:
            loss =loss.view(-1) #[680177]
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:int(loss.size()[0] * self.ohem_ratio)]

        return torch.mean(loss)

#unsupervised focal loss
class UnsupFocalLoss(nn.Module):
    def __init__(self, focal_gamma=1, focal_type=2, uda_softmax_temp=-1,uda_confidence_thresh=-1,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0 ,ohem=False, ohem_ratio=0.25, *args, **kwargs):
        super(UnsupFocalLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp
        self.gamma = focal_gamma
        self.focal_type = focal_type #1 for 1-p_2; 2 for p_hat-p_2
        self.ohem = ohem #default false
        self.ohem_ratio = ohem_ratio #default 0.25

    def forward(self, logits_before, logits_after, name=None):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)


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

        loss = torch.clamp(loss, min=MIN_EPSILON)

        if self.ohem:
            loss =loss.view(-1) #[680177]
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:int(loss.size()[0] * self.ohem_ratio)]

        return torch.mean(loss)

#threshold-adaptive unsupervised focal loss
class AdaptUnsupFocalLoss(nn.Module):
    def __init__(self, focal_gamma=1, uda_softmax_temp=-1,uda_confidence_thresh=0.8,weight=None,tsa=None, num_train_steps=-1,start_thresh=0.0,n_class = 19, adapt_b=0.2, adapt_a=0.9, adapt_d=8.0, a_decay=0,ohem=False, ohem_ratio=0.25, focal_type=1, *args, **kwargs):
        super(AdaptUnsupFocalLoss, self).__init__()
        self.weight = weight
        self.tsa = tsa
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.num_train_steps = num_train_steps
        self.start_thresh = start_thresh
        self.uda_confidence_thresh = torch.tensor(uda_confidence_thresh, dtype=torch.float).cuda()
        self.uda_softmax_temp = uda_softmax_temp
        self.focal_type = focal_type
        self.gamma = focal_gamma
        self.n_class = n_class
        self.cls_thresh = (torch.ones(n_class, dtype=torch.float64) * uda_confidence_thresh).cuda() #change np.ones to torch.ones (float64)
        self.adapt_b = adapt_b
        self.adapt_a = adapt_a #history parameter
        self.adapt_d = adapt_d
        self.a_decay = a_decay
        self.ohem = ohem #default false
        self.ohem_ratio = ohem_ratio #default 0.25

    def ias_thresh(self, preds_before_detach, n_class, adapt_b=0.8, w=None, adapt_d=8.0):
        logits_pred, label_pred = preds_before_detach.max(dim=1) #(N,H,W), (N,H,W)
        cls_thresh = (torch.ones(n_class, dtype=torch.float64)).cuda()
        for idx_cls in range(n_class):
            conf_tensor_list = logits_pred[label_pred==idx_cls]
            if len(conf_tensor_list) == 0:
                cls_thresh[idx_cls] = self.cls_thresh[idx_cls]
            else:
                conf_tensor_list_extend = torch.ones(conf_tensor_list.size()[0]+1, dtype=conf_tensor_list.dtype).cuda() * self.cls_thresh[idx_cls]
                conf_tensor_list_extend[:conf_tensor_list.size()[0]] = conf_tensor_list
                conf_tensor_list_extend,_ = torch.sort(conf_tensor_list_extend, descending=True)
                index = int(len(conf_tensor_list_extend) * adapt_b * self.cls_thresh[idx_cls] ** adapt_d)
                cls_thresh[idx_cls] = conf_tensor_list_extend[index]
        return cls_thresh

    def forward(self, logits_before, logits_after, name=None, iteration=10000):
        if self.uda_softmax_temp!=-1:
            logits_before = logits_before/self.uda_softmax_temp

        assert len(logits_before) == len(logits_after)

        preds_before=softmax(logits_before,dim=1)  #softmax is better
        preds_after=softmax(logits_after,dim=1)

        preds_before_detach = preds_before.detach()
        log_preds_before_detach = torch.log(preds_before_detach + LOG_EPSILON).detach()
        log_preds_after=torch.log(preds_after + LOG_EPSILON)

        if self.focal_type == 2:
            before_minus_after = preds_before_detach - preds_after
        elif self.focal_type == 1:
            before_minus_after = torch.ones_like(preds_before_detach) - preds_after
        _, max_index = preds_before_detach.max(dim=1)
        N,C,_,_ = preds_before_detach.size()
        select_mask = torch.zeros_like(preds_before_detach)
        select_mask.scatter_(1, max_index.unsqueeze(1), 1)
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
            
        #instance adaptive selector
        tmp_cls_thresh = self.ias_thresh(preds_before_detach, adapt_b=self.adapt_b, n_class=self.n_class, w=self.cls_thresh, adapt_d=self.adapt_d)
        if self.a_decay > 0:
            beta = 1 - self.adapt_a * math.exp(-iteration*self.a_decay)
            #1 - 0.9 * exp(-iter*0.01), gradually increase
        else:
            beta = self.adapt_a
        self.cls_thresh = beta * self.cls_thresh + (1 - beta) * tmp_cls_thresh
        self.cls_thresh[self.cls_thresh>=1] = 0.999

        #use the confidence threshold
        if self.uda_confidence_thresh!=-1:
            logit_amax, label = preds_before_detach.max(dim=1) #(N,H,W)
            label_cls_thresh = torch.ones_like(logit_amax, dtype=self.cls_thresh.dtype)
            for i in range(self.n_class):
                label_cls_thresh[label==i] = self.cls_thresh[i]
            loss_mask = logit_amax > label_cls_thresh
            adapt_ratio = float((torch.sum(loss_mask))) / (loss_mask.size()[0]*loss_mask.size()[1]*loss_mask.size()[2])
            loss_mask_solid = (logit_amax > float(self.uda_confidence_thresh))
            solid_ratio = float(torch.sum(loss_mask_solid)) / (loss_mask_solid.size()[0]*loss_mask_solid.size()[1]*loss_mask_solid.size()[2])
            loss=loss[loss_mask] #(N,H,W)
            
        loss = torch.clamp(loss, min=MIN_EPSILON)
        if self.ohem:
            loss =loss.view(-1) #[680177]
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:int(loss.size()[0] * self.ohem_ratio)]
        if self.uda_confidence_thresh!=-1:
            return torch.mean(loss), adapt_ratio, solid_ratio
        else:
            return torch.mean(loss)

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

