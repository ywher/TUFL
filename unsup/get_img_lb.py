import torch
import numpy as np
import random
import sys
import os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from utils.transformmasks import *
from utils.transformsgpu import oneMix
from utils.paste_utils import save_tensor_image, colorize_mask, strongTransform_class_mix, compute_edts_forPenalizedLoss

mix_img_save_pth = os.path.join(root_folder, 'outputs/images/mix') 

def get_source_img_lb(args, diter, train_epoch, sampler_trainset, trainloader):
    im, lb, im_origin, lb_origin, im_origin_norm = None, None, None, None, None
    try:
        if args.paste_mode != 'None':
            im, lb, _, im_origin, lb_origin, im_origin_norm = next(diter)
        else:
            im, lb, _ = next(diter)
        if not im.size()[0] == args.n_img_per_gpu_train: raise StopIteration
    except StopIteration:
        train_epoch += 1
        sampler_trainset.set_epoch(train_epoch)
        diter = iter(trainloader)
        if args.paste_mode != 'None':
            im, lb, _, im_origin, lb_origin, im_origin_norm = next(diter)
        else:
            im, lb, _ = next(diter)
    return im, lb, im_origin, lb_origin, im_origin_norm, diter, train_epoch, sampler_trainset


def get_unsup_single_img_lb(args, diter_unsup, unsup_epoch, sampler_unsupset, unsuploader):
    im_unsup, names_unsup, im_unsup_origin = None, None, None
    try:
        if args.paste_mode != 'None':
            im_unsup, names_unsup, im_unsup_origin = next(diter_unsup)
        else:
            im_unsup, names_unsup = next(diter_unsup)
        if not im_unsup.size()[0] == args.n_img_per_gpu_unsup: raise StopIteration
    except StopIteration:
        unsup_epoch += 1
        sampler_unsupset.set_epoch(unsup_epoch)
        diter_unsup = iter(unsuploader)
        if args.paste_mode != 'None':
            im_unsup, names_unsup, im_unsup_origin = next(diter_unsup)
        else:
            im_unsup, names_unsup = next(diter_unsup)
    return im_unsup, names_unsup, im_unsup_origin, diter_unsup, unsup_epoch, sampler_unsupset

def get_unsup_img_lb(args, diter_unsup, unsup_epoch, sampler_unsupset, unsuploader):
    im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm = \
        None, None, None, None, None, None
    try:
        #using the consistency regulation, augmented images
        if args.paste_mode != 'None':
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm = next(diter_unsup)
        else:
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup = next(diter_unsup)
        if not im_unsup.size()[0] == args.n_img_per_gpu_unsup: raise StopIteration
    except StopIteration:
        unsup_epoch += 1
        sampler_unsupset.set_epoch(unsup_epoch)
        diter_unsup = iter(unsuploader)
        if args.paste_mode != 'None':
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm = next(diter_unsup)
        else:
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup = next(diter_unsup)
    return im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, diter_unsup, unsup_epoch, sampler_unsupset

def get_unsup_pseudo_img_lb(args, diter_unsup, unsup_epoch, sampler_unsupset, unsuploader):
    im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, lb_unsup = \
        None, None, None, None, None, None, None
    try:
        #using the consistency regulation, augmented images
        if args.paste_mode != 'None':
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, lb_unsup = next(diter_unsup)
        else:
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup = next(diter_unsup)
        if not im_unsup.size()[0] == args.n_img_per_gpu_unsup: raise StopIteration
    except StopIteration:
        unsup_epoch += 1
        sampler_unsupset.set_epoch(unsup_epoch)
        diter_unsup = iter(unsuploader)
        if args.paste_mode != 'None':
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, lb_unsup = next(diter_unsup)
        else:
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup = next(diter_unsup)

    return im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, lb_unsup, diter_unsup, unsup_epoch, sampler_unsupset

def get_mix_img_lb(args, im_unsup_origin, im_unsup_origin_norm, pseudo_net, 
                    lb_unsup, lb_origin, im_origin, pick_class_function, criteria_unsup, unsupset, 
                    class_to_select, lt_cls_mixer, gaussian_blur, max_pool, to_pil, to_tensor, to_normalize):
    im_unsup_origin = im_unsup_origin.cuda()
    im_unsup_origin_norm = im_unsup_origin_norm.cuda()
    im_unsup_origin_pred,_,_,_ = pseudo_net(im_unsup_origin_norm) #(1,16,760,1280)
    pseudo_label = torch.softmax(im_unsup_origin_pred.detach(), dim=1) #(1,16,760,1280)
    max_probs, targets_u_w = torch.max(pseudo_label, dim=1) #(1,760,1280), (1,760,1280)
    if args.supervision_mode == 'unsup_pseudo':
        targets_u_w = lb_unsup.squeeze(1).cuda() #(1,1,760,1280) #use the pseudo label offline generated
    
    #origin source image and label
    lb_origin = lb_origin.cuda().long() #(1,1,760,1280)
    lb_origin = lb_origin.squeeze(1) #(1,760,1280)
    im_origin = im_origin.cuda() #(1,3,760,1280)
    
    all_classes = torch.unique(lb_origin)
    img_classes = all_classes.shape[0]
    
    #whether to use long tail paste
    if args.long_tail:
        long_tail_paste = True if np.random.uniform() < args.long_tail_p else False

    if args.long_tail and long_tail_paste:
        classes = (all_classes[torch.Tensor(np.random.choice(img_classes, max(int((img_classes+img_classes%2)/2)-2, 1), replace=False)).long()]).cuda() #(N//2-2), 
    else:    
        classes = (all_classes[torch.Tensor(np.random.choice(img_classes, int((img_classes+img_classes%2)/2),replace=False)).long()]).cuda() #(N), 
    if args.class_relation:
        update_classes = pick_class_function.get_pick_classes(list(all_classes.cpu().numpy()),list(classes.cpu().numpy()))
        classes = torch.Tensor(update_classes).long().cuda()
    mix_mask = generate_class_mask(lb_origin.squeeze(0), classes).unsqueeze(0).cuda().float() #(1,760,1280)
    if args.long_tail and long_tail_paste:
        if args.unsup_loss == 'adapt_focal':
            select_lt_number = random.sample(range(2,5),1)[0]
            select_p = torch.softmax(1/(criteria_unsup.cls_thresh), 0).cpu().numpy()
            select_classes = list(np.random.choice(class_to_select, select_lt_number, replace=True, p=select_p))
            im_origin, lb_origin, mix_mask = lt_cls_mixer.mix(im_origin, lb_origin, mix_mask, select_classes) #(1,3,H,W), (1,H,W), (1,H,W)
        else:    
            im_origin, lb_origin, mix_mask = lt_cls_mixer.mix(im_origin, lb_origin, mix_mask, random.sample(class_to_select, 2)) #(1,3,H,W), (1,H,W), (1,H,W)
    mix_img, _ = oneMix(mix_mask, data=torch.cat((im_origin, im_unsup_origin))) #(1,3,760,1280)
    _, mix_lb = oneMix(mix_mask.long(), target=torch.cat((lb_origin, targets_u_w))) #(1,760,1280)
    if args.mixed_gaussian_kernel > 0:
        mix_img_blur = gaussian_blur(mix_img)
    
    mixed_weight = torch.sum(max_probs.ge(0.968).long(
    ) == 1).item() / np.size(np.array(mix_lb.cpu()))
    pixelWiseWeight = mixed_weight * torch.ones(max_probs.shape).cuda() #(1,760,1280)
    onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()  #(1,760,1280)
    _, pixelWiseWeight = oneMix(mix_mask, target=torch.cat((onesWeights,pixelWiseWeight))) #(1,760,1280)
    pixelWiseWeight = pixelWiseWeight.cuda() #(1,760,1280)
    if args.mixed_boundary:
        mix_boundary = max_pool(mix_mask) + max_pool(-mix_mask)
        if args.mixed_gaussian_kernel > 0:
            mix_img, _ = oneMix(mix_boundary, data=torch.cat((mix_img_blur, mix_img)))
        pixelWiseWeight[mix_boundary==1] *= 2
    elif args.bapa_boundary:
        boundary_weight = compute_edts_forPenalizedLoss(mix_mask, args.bapa_boundary) #numpy
        boundary_weight = torch.from_numpy(boundary_weight).cuda()
        pixelWiseWeight += boundary_weight
    #augment the mix image
    pil_mix_img = to_pil(mix_img.detach().squeeze(0).cpu())
    mix_img = to_tensor(unsupset.trans_unsup_color(pil_mix_img)).unsqueeze(0)
    del pil_mix_img
    mix_img = to_normalize((mix_img.squeeze(0))).unsqueeze(0) #(1,3,720,1280)
    return mix_img, mix_lb, pixelWiseWeight

def get_dsp_mix_img_lb(args, im_unsup_origin, im_unsup_origin_norm, pseudo_net, 
    image_mix_origin, label_mix_origin, lb_unsup, lb_origin, im_origin, class_to_select, lt_cls_mixer, to_normalize):
    #source image from next iter, use its contents to paste
    image_mix_origin = image_mix_origin.cuda() #(1,3,H,W)
    label_mix_origin = label_mix_origin.cuda().long() #(1,1,H,W)
    label_mix_origin = label_mix_origin.squeeze(1) #(1,H,W)
    #target iamge
    im_unsup_origin = im_unsup_origin.cuda()
    im_unsup_origin_norm = im_unsup_origin_norm.cuda()
    im_unsup_origin_pred,_,_,_ = pseudo_net(im_unsup_origin_norm) #(1,16,760,1280)
    pseudo_label = torch.softmax(im_unsup_origin_pred.detach(), dim=1) #(1,16,760,1280)
    max_probs, targets_u_w = torch.max(pseudo_label, dim=1) #(1,760,1280), (1,760,1280)
    if args.supervision_mode == 'unsup_pseudo':
        targets_u_w = lb_unsup.squeeze(1).cuda() #(1,1,760,1280) #use the pseudo label offline generated
    
    #origin source image and label from last iter
    lb_origin = lb_origin.cuda().long() #(1,1,760,1280)
    lb_origin = lb_origin.squeeze(1) #(1,760,1280)
    im_origin = im_origin.cuda() #(1,3,760,1280)
    cls_to_use = random.sample(class_to_select, 2) #long tail class
    all_classes = torch.unique(label_mix_origin)
    img_classes = all_classes.shape[0]

    #whether to use long tail paste

    classes = (all_classes[torch.Tensor(np.random.choice(img_classes, max(int((img_classes+img_classes%2)/2), 1), replace=False)).long()]).cuda() #(N//2-2), 
    mix_mask = generate_class_mask(label_mix_origin.squeeze(0), classes).unsqueeze(0).cuda().float() #(1,760,1280)
    
    source_mix_img, source_mix_lbl, target_mix_img, target_mix_lbl, MixMask_lam = \
                strongTransform_class_mix(image_mix_origin, im_origin, im_unsup_origin, label_mix_origin, lb_origin, targets_u_w,
                                          mix_mask, lt_cls_mixer, cls_to_use, mixWeight=args.soft_weight)
    source_mix_img = to_normalize((source_mix_img.squeeze(0))).unsqueeze(0) #(1,3,H,W)
    target_mix_img = to_normalize((target_mix_img.squeeze(0))).unsqueeze(0) #(1,3,H,W)
    
    mixed_weight = torch.sum(max_probs.ge(0.968).long(
    ) == 1).item() / np.size(np.array(target_mix_lbl.cpu()))
    pixelWiseWeight = mixed_weight * torch.ones(max_probs.shape).cuda() #(1,760,1280)
    onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()  #(1,760,1280)
    _, pixelWiseWeight = oneMix(mix_mask, target=torch.cat((onesWeights,pixelWiseWeight))) #(1,760,1280)
    pixelWiseWeight = pixelWiseWeight.cuda() #(1,760,1280)
    #augment the mix image
    return source_mix_img, source_mix_lbl, lb_origin, target_mix_img, target_mix_lbl, targets_u_w, pixelWiseWeight
