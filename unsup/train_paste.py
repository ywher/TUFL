#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
import os
import os.path as osp
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
output_dir= os.path.join(root_folder, 'outputs')

from model import BiSeNet
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.transforms.functional import to_tensor
from tensorboardX import SummaryWriter

import logging
from unsup.parse_args import parse_args ##config for training
from unsup.get_folder_name import get_folder_name ##get the foler name to store result
from unsup.get_dataset import get_source_dataset, get_taret_dataset
from unsup.load_pretrained import load_pretrained_model
from unsup.get_optim import get_optim
from unsup.get_loss import get_loss
from unsup.get_func import get_func
from unsup.get_img_lb import get_source_img_lb, get_unsup_single_img_lb, get_unsup_img_lb, get_unsup_pseudo_img_lb, get_mix_img_lb, get_dsp_mix_img_lb
from unsup.train_step import train_step



def train():
    logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='log.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志.#a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s') #日志格式
    logger = logging.getLogger()
    ###config parameters for training
    args = parse_args()
    ###result path
    folder_name = get_folder_name(args) 
    respth = os.path.join(output_dir, folder_name) #result path
    logger.info("respth "+respth)
    if not osp.exists(respth):
        os.makedirs(respth,exist_ok=True)
    ###distributed training on multi GPUs
    torch.cuda.set_device(args.local_rank)
    ###set the communication between 2 gpus
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:33271',
                world_size = torch.cuda.device_count(),
                rank=args.local_rank
                )
    ###log the config parameters
    logger.info(args)

    ###get the source, target and validation dataset
    # define the dataset in source domain and initialize the trainset for supervised learning
    if args.dataset:
        n_classes, crop_size, dataroot, trainset, sampler_trainset, trainloader = get_source_dataset(args)
        logger.info("Number of train set: "+ str(len(trainset))) #2975 for Cityscapes
    #load unsupervised training set in target domain
    if args.supervision_mode != 'sup':
        unsupset, sampler_unsupset, unsuploader = get_taret_dataset(args, crop_size, n_classes)
        logger.info("Number of unsup set: "+str (len(unsupset))) #3200 for CrossCity
    logger.info("crop size is "+str (crop_size))
    ### initialize model
    if args.segmentation_model == 'BiSeNet':
        net = BiSeNet(n_classes=n_classes, sup_mode=args.sup_mode)
        pseudo_net = BiSeNet(n_classes=n_classes, sup_mode=args.sup_mode)
    #load pretrained model
    if args.pretrained_dataset is not None:
        net, pseudo_net, logger = load_pretrained_model(args, net, pseudo_net, logger)
    #put the model from cpu to gpu
    net.cuda()
    if args.freeze_bn:
        net.eval() #for freezing the batch normalization layer
    else:
        net.train() #for train mode, the bn layer is not freezed
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank)
    
    pseudo_net.cuda()
    pseudo_net.eval() #only update parameters except bn layer
    pseudo_net = nn.parallel.DistributedDataParallel(pseudo_net,
        device_ids = [args.local_rank, ],
        output_device = args.local_rank)
    
    ###loss function###
    #criteria_p: supervised loss; criteria_unsup: unsupervised loss; criteria_paste: loss for CIM
    criteria_p, criteria_unsup, criteria_paste = get_loss(args, crop_size, n_classes)
    ## optimizer
    optim = get_optim(args, net)

    ## train loop
    ###get data_iter
    diter = iter(trainloader)
    train_epoch = 0
    if args.supervision_mode!='sup':
        diter_unsup = iter(unsuploader)
        unsup_epoch = 0
    ###tensorboard
    if args.tensorboard:
        tensorboard_path = osp.join(respth, 'tensorbaord')
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        writer = SummaryWriter(tensorboard_path)

    #initialize the training step class
    training_step = train_step(args=args, criteria_p=criteria_p, criteria_unsup=criteria_unsup, \
            criteria_paste=criteria_paste, n_classes=n_classes)
    ###get some fucntions
    global to_pil
    to_normalize, to_pil, gaussian_blur, max_pool, \
    lt_cls_mixer, class_to_select, pick_class_function = get_func(args, dataroot, crop_size)
    for it in range(args.max_iter):
        #get source image and label
        im, lb, im_origin, lb_origin, _, diter, train_epoch, sampler_trainset = \
            get_source_img_lb(args, diter, train_epoch, sampler_trainset, trainloader)
        #get unsupervised images and validataion images and labels
        lb_unsup = None
        if args.supervision_mode == 'unsup_single':
            im_unsup, names_unsup, im_unsup_origin, diter_unsup, unsup_epoch, sampler_unsupset = \
                get_unsup_single_img_lb(args, diter_unsup, unsup_epoch, sampler_unsupset, unsuploader)
        elif args.supervision_mode == 'unsup':
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, diter_unsup, unsup_epoch, sampler_unsupset = \
                get_unsup_img_lb(args, diter_unsup, unsup_epoch, sampler_unsupset, unsuploader)
        elif args.supervision_mode == 'unsup_pseudo':
            im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, im_unsup_origin, im_unsup_origin_norm, lb_unsup, diter_unsup, unsup_epoch, sampler_unsupset = \
                get_unsup_pseudo_img_lb(args, diter_unsup, unsup_epoch, sampler_unsupset, unsuploader)
        #get mixed image and label
        mix_img, mix_lb, pixelWiseWeight = None, None, None
        source_mix_img, source_mix_lbl, target_mix_img, target_mix_lbl, targets_u_w = None, None, None, None, None
        if args.paste_mode == 'Single':
            mix_img, mix_lb, pixelWiseWeight = get_mix_img_lb(args, im_unsup_origin, im_unsup_origin_norm, pseudo_net, 
                lb_unsup, lb_origin, im_origin, pick_class_function, criteria_unsup, unsupset, 
                class_to_select, lt_cls_mixer, gaussian_blur, max_pool, to_pil, to_tensor, to_normalize)
        elif args.paste_mode == 'Dual_soft':
            _, _, image_mix_origin, label_mix_origin, _, diter, train_epoch, sampler_trainset = \
            get_source_img_lb(args, diter, train_epoch, sampler_trainset, trainloader)
            source_mix_img, source_mix_lbl, lb_origin, target_mix_img, target_mix_lbl, targets_u_w, pixelWiseWeight = \
            get_dsp_mix_img_lb(args, im_unsup_origin, im_unsup_origin_norm, pseudo_net, 
            image_mix_origin, label_mix_origin, lb_unsup, lb_origin, im_origin, class_to_select, lt_cls_mixer, to_normalize)
        if args.supervision_mode=='sup':
            loss, optim, net = training_step.sup_step(im, lb, optim, net)
        # semi supervised
        elif args.supervision_mode == 'unsup_single':
            loss, optim, net = training_step.unsup_single_step(im, lb, im_unsup, optim, net)
        #unsupervised
        elif args.supervision_mode=='unsup' or args.supervision_mode == 'unsup_pseudo':
            loss, optim, net, pseudo_net= training_step.unsup_step(im, lb, im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, it, \
                        mix_img, mix_lb, source_mix_img, source_mix_lbl, lb_origin, target_mix_img, target_mix_lbl, targets_u_w, pixelWiseWeight, optim, net, pseudo_net)
        
        loss.backward()
        optim.step()

        ## logger.info training log message
        if (it+1)%args.msg_iter==0:
            writer, logger = training_step.msg_iter(writer, logger, it, optim)
        #dump the final model
        if (it + 1) % (int(0.1*args.max_iter)) == 0:
            ## dump the final model
            save_pth = osp.join(respth, 'model_epoch_'+str(it+1)+'.pth')
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank()==0: torch.save(state, save_pth)
            logger.info('Model saved to: {}'.format(save_pth))

if __name__ == "__main__":
    # print('get in main')
    train()
