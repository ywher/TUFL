#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from datasets.cityscapes import CityScapes
from datasets.crosscity import CrossCity
from unsup.model import BiSeNet

import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import numpy as np
from tqdm import tqdm
import math
import copy
import json
from PIL import Image
import argparse
from collections import OrderedDict
import matplotlib.cm as mpl_color_map

def apply_colormap(activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)

    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    return no_trans_heatmap

def grey2color(image,color_map):
    height, width = image.shape[:2]
    color_label = np.zeros((height, width, 3), np.uint8)
    image2=copy.deepcopy(image)
    unique= np.unique(image2, return_counts=True)
    unique=list(unique[0])

    for grayscale_color in unique:
        if grayscale_color==255:
            pixel_rgb=[255,255,255]
        else:
            pixel_rgb = color_map[grayscale_color]

        mask = (image2 == grayscale_color)
        color_label[mask] = pixel_rgb

    color_label = color_label.astype(np.uint8)
    return color_label

def grey2id(image_trainid,id_map):
    image = copy.deepcopy(image_trainid)
    unique = np.unique(image, return_counts=True)
    unique = list(unique[0])
    unique.sort(reverse=True)

    for train_id in unique:
        id = id_map[train_id]
        image[image == train_id] = id

    image = image.astype(np.uint8)
    return image
    
class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],#multi scale testing
            n_classes = 19,
            lb_ignore = 255,
            cropsize = 1024,#cityscape 1024
            flip = True,simple_mode=False,save_pic=None, config=None,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model
        self.save_pic=save_pic
        self.img_count = 0
        if simple_mode:
            self.scales=[1]
            self.flip=False

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]
    
    def eval_whole(self,image,return_features=False):
        with torch.no_grad():
            out, _, _, features = self.net(image)
            prob = F.softmax(out, 1)
        if return_features:
            return prob, features
        else:
            return prob

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5/6.
        N, C, H, W = im.size()
        long_size, short_size = (H,W) if H>W else (W,H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize*stride_rate) #1024 * 5 / 6 = 853.33
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size() #(3,1024,2048)
            n_x = math.ceil((W-cropsize)/stride)+1 #ceil(1.2)=2+1=3 times (0,853,1024)
            n_y = math.ceil((H-cropsize)/stride)+1 #1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize) #1024, 1024
                    hst, wst = hed-cropsize, wed-cropsize #0,0
                    chip = im[:, :, hst:hed, wst:wed]#(1024,1024)
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob

    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)

        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def eval_whole_img(self,img):
        with torch.no_grad():
            out = self.net(img)[0]
            prob = F.softmax(out, 1)
        return prob
    
    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        #initialize the adapt class
        for i, (imgs, label,imgs_name) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc) #after softmax, multi-crop eval, may greater than 1
                probs += prob.detach().cpu()

            _, C, _, _ = probs.shape

            probs = probs.data.numpy()
            #here probs are not in the probability form due to the exponential calculation
            preds = np.argmax(probs, axis=1) #(1,C,H,W) -> (1,H,W), label

            label=label.data.numpy().squeeze(1)

            if self.save_pic:
                id_dir=os.path.join(self.save_pic,'preds_label_id')
                gt_dir=os.path.join(self.save_pic,'gt')
                color_dir = os.path.join(self.save_pic, 'preds_color')
                entropy_dir = os.path.join(self.save_pic, 'entropy')
                if not os.path.exists(id_dir):
                    os.makedirs(id_dir)
                if not os.path.exists(gt_dir):
                    os.makedirs(gt_dir)
                if not os.path.exists(color_dir):
                    os.makedirs(color_dir)
                if not os.path.exists(entropy_dir):
                    os.makedirs(entropy_dir)

                for j in range(preds.shape[0]):

                    label_color = grey2color(label[j, :, :],color_map)
                    label_pil = Image.fromarray(label_color)
                    label_pil.save(gt_dir+'/'+imgs_name[j]+'.png')

                    preds_id= preds[j, :, :].astype(np.uint8)
                    preds_id_pil = Image.fromarray(preds_id)
                    preds_id_pil.save(id_dir+'/'+imgs_name[j]+'.png')

                    preds_color = grey2color(preds[j, :, :],color_map)
                    preds_pil=Image.fromarray(preds_color)
                    preds_pil.save(color_dir+'/'+imgs_name[j]+'.png')

                    # entropy output
                    img_tensor = imgs[j, :, :, :].unsqueeze(0).cuda()
                    out = self.net(img_tensor)[0]
                    N, C, H, W = out.size()
                    probs_softmax = torch.softmax(out, dim=1)
                    probs_softmax = probs_softmax.detach().cpu().data.numpy()
                    output_ent = np.sum(-np.multiply(probs_softmax, np.log(probs_softmax)), axis=1,
                                        keepdims=False) / np.log(C)
                    output_ent = (np.squeeze(output_ent) * 255).astype(np.uint8)

                    output_ent_color = apply_colormap(output_ent, 'viridis')
                    output_ent_color = output_ent_color.convert("RGB")

                    output_ent_color.save(entropy_dir + '/' + imgs_name[j] + '.png')

            hist_once = self.compute_hist(preds,label )
            hist = hist + hist_once
        
        num = np.diag(hist)
        den = np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist)
        IOUs = num / den
        mIOU = np.mean(IOUs)
        
        return IOUs,mIOU

    def test(self):
        id_dir = os.path.join(self.save_pic, 'pred_id')
        color_dir = os.path.join(self.save_pic, 'pred_color')
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)
        else:
            shutil.rmtree(id_dir)
            os.makedirs(id_dir)

        if not os.path.exists(color_dir):
            os.makedirs(color_dir)
        else:
            shutil.rmtree(color_dir)
            os.makedirs(color_dir)


        ## evaluate
        n_classes = self.n_classes
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, imgs_name) in enumerate(dloader):
            N, C, H, W = imgs.size()
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            if self.save_pic:
                for j in range(preds.shape[0]):
                    preds_id= grey2id(preds[j, :, :],id_map)
                    preds_id_pil = Image.fromarray(preds_id)
                    preds_id_pil.save(id_dir+'/'+imgs_name[j]+'.png')

                    # print(preds_id_pil.mode)

                    preds_color = grey2color(preds[j, :, :],color_map)
                    preds_pil=Image.fromarray(preds_color)
                    preds_pil.save(color_dir+'/'+imgs_name[j]+'.png')

def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--dataset', dest='dataset', type=str, default='CrossCity', choices=['CityScapes', 'CrossCity'], help='evaluate dataset')
    parse.add_argument('--iou_mode', dest='iou_mode', type=str, default='cross', choices=['except_background','synthia',None,'cross'], help='iou classes number')
    parse.add_argument('--dataset_mode', dest='dataset_mode', type=str, default='test', choices=['train','val','test','unsup_single'], help='validation split')
    parse.add_argument('--city', dest='city', type=str, default='Rome', choices=['all', 'Rome', 'Rio', 'Tokyo', 'Taipei'], help='target city')
    parse.add_argument('--batchsize', dest='batchsize', type=int, default=1, help='validation batch size')
    parse.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parse.add_argument('--simple_mode', dest='simple_mode', action='store_true', help='simple evaluation (whole image)')
    parse.add_argument('--save_pic', dest='save_pic', type=str, default=None, help='the path to save the prediction result')
    parse.add_argument('--save_pth',dest = 'save_pth', type = str, default = '', help='the path to load the trained model')
    parse.add_argument('--segmentation_model', dest='segmentation_model', type=str, default='BiSeNet', choices=['BiSeNet'])
    return parse.parse_args()

def evaluate():
    args=parse_args()
    mode=args.dataset_mode #default val
    city = args.city
    dataset_name = args.dataset #default CrossCity
    simple_mode = args.simple_mode #action=store true
    save_pic = args.save_pic #the way to save the pictures
    save_pth = os.path.join(root_folder, 'outputs', args.save_pth) #the way to laod the model
    seg_model = args.segmentation_model
    ## dataset
    batchsize = args.batchsize
    n_workers = args.num_workers
    print('\n')
    print('===='*20)
    print('evaluating the model ...')

    if dataset_name=='CityScapes':
        dspth= os.path.join(root_folder, 'data/cityscapes') 
        cropsize = 1024
        if args.iou_mode == "synthia":
            n_classes = 16
        elif args.iou_mode == 'cross':
            n_classes = 13
        else:
            n_classes = 19
        if mode=='train':
            dsval = CityScapes(dspth, mode=mode, if_augmented=False, n_class=n_classes, data_ratio=0.2)
        else:
            dsval = CityScapes(dspth, mode=mode, if_augmented=False, n_class=n_classes) #val
    elif dataset_name=='CrossCity':
        cropsize = 647
        dspth = os.path.join(root_folder, 'data/NTHU')
        n_classes = 13
        dsval = CrossCity(dspth, mode='test', if_augmented=False, n_class=n_classes, city=city)
        print('city:', city, ', len of data:', len(dsval))

    global color_map,id_map
    if dataset_name == 'CrossCity':
        fr = open('./datasets/crosscity_info.json', 'r')
    elif dataset_name=='CityScapes':
        fr = open('./datasets/cityscapes_info.json', 'r')
    labels_info = json.load(fr)
    #global color_map,id_map
    color_map = {el['trainId']: el['color'] for el in labels_info}
    id_map = {el['trainId']: el['id'] for el in labels_info}  # if el['ignoreInEval']==False
    fr.close()
    if seg_model == 'BiSeNet':
        net = BiSeNet(n_classes=n_classes)

    single_dict=torch.load(save_pth)
    net.load_state_dict(single_dict, strict = True)
    net.cuda()
    net.eval()
    
    print("model path ", save_pth)
    print("dataset path ", dspth)
    print("validation dataset:",str(mode),', ',end='')
    print("simple mode:",str(simple_mode))

    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    ## evaluator
    print('compute the mIOU')
    evaluator = MscEval(net, dl, n_classes=n_classes, cropsize=cropsize, simple_mode=simple_mode, 
    save_pic=save_pic, config=args)

    ## eval
    if (mode=='test' or mode == 'unsup_single') and dataset_name=='CityScapes':
        # test
        evaluator.test()
    else:
        IOUs,mIOU = evaluator.evaluate()

        if dataset_name=='CityScapes' or dataset_name == 'CrossCity':
            if args.iou_mode == "synthia":
                train_id_list = ['road', 'sidew', 'build', 'wall', 'fence', 'pole', 'tligh',
                                 'tsign', 'veget', 'sky', 'perso', 'rider', 'car', 'bus',  'motor', 'bike']
            elif args.iou_mode == 'cross':
                train_id_list = ['road', 'sidew', 'build', 'tligh', 'tsign', 'veget', 
                                 'sky', 'perso', 'rider', 'car', 'bus',  'motor', 'bike']
            else:
                train_id_list=['road','sidew','build','wall','fence','pole','tligh','tsign','veget','terra',
                           'sky','perso','rider','car','truck','bus','train','motor','bike']

            iou_dict = OrderedDict()
            for i in range(len(train_id_list)):
                iou_dict[train_id_list[i]] = (IOUs[i])

            if args.iou_mode == "except_background":
                del iou_dict['road']
                print('mIOU excluding background is',end=" ")
            elif args.iou_mode == "synthia":
                print('mIOU for 16 classes is',end=" ")
                mIOU = np.average(list(iou_dict.values()))
                print(mIOU)
                print()
                for key, value in iou_dict.items():
                    print(key, '\t', end=' ')
                print('Average')
                print("")
                for key, value in iou_dict.items():
                    print('%.3f'%value, '\t', end=' ')
                print(mIOU)
                print("")

                del iou_dict['wall']
                del iou_dict['fence']
                del iou_dict['pole']
                print('mIoU for 13 classes is', end=" ")
            else:
                print("mIoU for all is ",end=" ")

            mIOU = np.average(list(iou_dict.values()))
            print(mIOU)

            for key, value in iou_dict.items():
                print(key, '\t', end=' ')
            print('Average')
            print("")
            for key, value in iou_dict.items():
                print('%.3f'%value, '\t', end=' ')
            print(mIOU)
            print("")
        else:
            print('mIOU is: {:.6f}'.format(mIOU))
            print('Seperate IOU is: {}'.format(IOUs))

if __name__ == "__main__":
    evaluate()
    # realtime_evaluate()

