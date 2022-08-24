import sys, os
import torch
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from utils.loss import *
from utils.class_freqency import cfg

def get_loss(args, crop_size, n_classes):
    criteria_p, criteria_unsup, criteria_paste = None, None, None
    #the weight for each class
    class_weight=torch.FloatTensor(cfg.CLASS_WEIGHT['CityScapes'+'_'+str(n_classes)][args.weight_mode]).cuda()
    ignore_idx = args.ignore_index
    if args.sup_loss == 'regular': #default regular weighted cross entropy loss
        if args.dataset == 'Mapillary':
            criteria_p = RegularCrossLoss(ignore_lb=ignore_idx)
        else:
            criteria_p = RegularCrossLoss(ignore_lb=ignore_idx, weight=class_weight)
    elif args.sup_loss == 'ohem':
        score_thres = 0.7
        n_min = args.n_img_per_gpu_train * crop_size[0] * crop_size[1] // 16
        if args.dataset == 'Mapillary':
            criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        else:
            criteria_p = OhemCELoss(weight=class_weight, thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    
    #unsupervised loss
    if args.supervision_mode == 'unsup_single':
        if args.unsup_single_loss=='entropy':
            criteria_unsup = MinimumEntropyLoss(weight=class_weight,uda_confidence_thresh=args.uda_confidence_thresh,uda_softmax_temp=args.uda_softmax_temp, ohem=args.loss_ohem, ohem_ratio=args.loss_ohem_ratio)
        elif args.unsup_single_loss=='square':
            criteria_unsup = MinimumSquareLoss(weight=class_weight, uda_confidence_thresh=args.uda_confidence_thresh,uda_softmax_temp=args.uda_softmax_temp) #defualt=-1
        elif args.unsup_single_loss=='confi_entropy':
            criteria_unsup = ConfidenceEntropy(weight=class_weight,uda_confidence_thresh=args.uda_confidence_thresh,uda_softmax_temp=args.uda_softmax_temp)
    elif args.supervision_mode in ['unsup','unsup_both','unsup_pseudo']:
        if args.unsup_loss =='crossentropy':
            criteria_unsup = UnsupCrossEntropyLoss(weight=class_weight,
                                                   uda_confidence_thresh=args.uda_confidence_thresh,
                                                   uda_softmax_temp=args.uda_softmax_temp, ohem=args.loss_ohem, ohem_ratio=args.loss_ohem_ratio)
        elif args.unsup_loss =='focal':
            criteria_unsup = UnsupFocalLoss(weight=class_weight,
                                            uda_confidence_thresh=args.uda_confidence_thresh,
                                            uda_softmax_temp=args.uda_softmax_temp,
                                            focal_gamma=args.focal_gamma)
        elif args.unsup_loss == 'adapt_focal':
            criteria_unsup = AdaptUnsupFocalLoss(weight=class_weight,
                                            uda_confidence_thresh=args.uda_confidence_thresh,
                                            uda_softmax_temp=args.uda_softmax_temp,
                                            focal_gamma=args.focal_gamma,
                                            adapt_b=args.adapt_b,
                                            adapt_a=args.adapt_a,
                                            adapt_d=args.adapt_d,
                                            a_decay=args.a_decay,
                                            n_class=n_classes)
    if args.paste_mode != 'None':
        criteria_paste = CrossEntropyLoss2dPixelWiseWeighted(weight=class_weight,ignore_index=ignore_idx)
    return criteria_p, criteria_unsup, criteria_paste