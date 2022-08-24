#!/usr/bin/python
# -*- encoding: utf-8 -*-


from datasets.transform_paste import Compose_P, HorizontalFlip_P
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from PIL import Image
import numpy as np
from .transform import *
from .transform_image import *
from .transform_paste import *

class CityScapes(Dataset):
    #in the order of height ,width
    def __init__(self, rootpth, cropsize=(512, 1024), mode='train', if_augmented=True, data_ratio=1.0, resize=(1.0,1.0), 
                n_class=19, pseudo_dir = None, keep_origin=False, *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'train_Rome', 'train_Rio', 'train_Tokyo', 'train_Taipei', 'val', 'test', 'unsup','unsup_single')
        self.label_list=['train', 'train_Rome', 'train_Rio', 'train_Tokyo', 'train_Taipei', 'val']
        self.mode = mode
        self.ignore_lb = 255
        self.affine_kwargs = {"min_rot": -30.0, "max_rot": 30.0,
                              "min_shear": -5.0, "max_shear": 5.0,
                              "min_scale": 0.8, "max_scale": 1.2} #affine augmentation
        self.if_augmented = if_augmented
        self.resize = resize #resizing coefficient for height and width respectively
        cropsize=(cropsize[1],cropsize[0]) #change from (h,w) to (w,h), (1280,760)
        self.cropsize = cropsize

        #from labelId to trainId
        if n_class==19: #for Cityscapes
            self.lb_map = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18}
        elif n_class==16: #remove 22,27,31, terrain, truck, train
            self.lb_map = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,23:9,24:10,25:11,26:12,28:13,32:14,33:15}
        elif n_class == 13: #remove 12,13,17 wall, fence, pole
            self.lb_map = {7:0,8:1,11:2,19:3,20:4,21:5,23:6,24:7,25:8,26:9,28:10,32:11,33:12}
        self.keep_origin = keep_origin #whether return original images and labels without augmentation, in paste mode
        
        ## parse img directory, get images names and pathes
        folder_name=mode
        if self.mode == 'unsup':
            folder_name='unsup_single' #19997
        self.pseudo_dir = pseudo_dir #where to load the pseudo labels
        impth = osp.join(rootpth, 'leftImg8bit', folder_name)
        folders = os.listdir(impth)

        self.imgs = {}
        imgnames = []
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))
        self.imnames = imgnames

        ## parse gt directory, get label names and pathes
        if self.mode in self.label_list:
            self.labels = {}
            gtnames = []
            if folder_name in ['train_Rome', 'train_Rio', 'train_Tokyo', 'train_Taipei']:
                folder_name = 'train' #share the same labels
            if self.pseudo_dir: #for pseudo label self-traing
                pseudo_lbnames = os.listdir(self.pseudo_dir)
                pseudo_names = [el.replace('_pseudo_label.png', '') for el in pseudo_lbnames if '_pseudo_label.png' in el]  ###need attention
                pseudo_lbpths = [osp.join(self.pseudo_dir, el) for el in pseudo_lbnames if '_pseudo_label.png' in el]
                gtnames.extend(pseudo_names)
                self.labels.update(dict(zip(pseudo_names, pseudo_lbpths)))
            else:
                gtpth = osp.join(rootpth, 'gtFine', folder_name)
                folders = os.listdir(gtpth)
                for fd in folders:
                    fdpth = osp.join(gtpth, fd)
                    lbnames = os.listdir(fdpth)
                    lbnames = [el for el in lbnames if 'labelIds' in el]
                    names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
                    lbpths = [osp.join(fdpth, el) for el in lbnames]
                    gtnames.extend(names)
                    self.labels.update(dict(zip(names, lbpths)))

            assert set(imgnames) == set(gtnames)
            assert set(self.imnames) == set(self.imgs.keys())
            assert set(self.imnames) == set(self.labels.keys())

        data_step=int(1.0/data_ratio)
        self.imnames.sort()
        imnames=[]
        for i, imname in enumerate(self.imnames):
            if i%data_step==0:
                imnames.append(imname)
        self.imnames=imnames
        
        self.len = len(self.imnames)

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.to_tensor_only = transforms.ToTensor()
        #augment the source domain image and label pairs
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5)),
            RandomCrop(cropsize),
            Shadow(shadow=(0.01, 0.3),
                   shadow_file=os.path.join(dir_path, "shadow_pattern.jpg"),
                   shadow_crop_range=(0.02, 0.5)),
            Noise(noise=4),
            Blur(blur=0.2),
        ])
        #augment the mixed image and label pairs
        self.trans_train_paste = Compose_P([
            ColorJitter_P(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip_P(),
            RandomScale_P((0.75, 1.0, 1.25, 1.5)),
            RandomCrop_P(cropsize),
            Shadow_P(shadow=(0.01, 0.3),
                   shadow_file=os.path.join(dir_path, "shadow_pattern.jpg"),
                   shadow_crop_range=(0.02, 0.5)),
            Noise_P(noise=4),
            Blur_P(blur=0.2),
        ])
        self.random_crop = Compose([RandomCrop(cropsize)])
        #augment the target domain image
        self.trans_unsup_image = Compose([
            HorizontalFlip_image(),
            RandomScale_image((0.75, 1.0, 1.25, 1.5)),
            RandomCrop_image(cropsize)
        ])
        self.randcrop = RandomCrop_image(self.cropsize)
        self.flip = HorizontalFlip_image()
        #augment the target domain image, texture transformation
        self.trans_unsup_color = Compose([
            ColorJitter_image(
                brightness=[0.4, 1.2],
                contrast=[0.4, 2],
                saturation=[0.5, 2],
                sharpeness=[0.4, 2]),
            Shadow_image(shadow=(0.01, 0.3),
                         shadow_file=os.path.join(dir_path, "shadow_pattern.jpg"),
                         shadow_crop_range=(0.02, 0.5)),
            Noise_image(noise=[3, 5]),
            Blur_image(blur=0.2),
            Style_image(p=0.3)])

    def __getitem__(self, idx):
        # print('self.mode', self.mode)
        if self.mode in self.label_list: #train/val mode with label
            fn = self.imnames[idx]
            impth = self.imgs[fn]
            lbpth = self.labels[fn]

            img = Image.open(impth)
            label = Image.open(lbpth)

            assert img.size == label.size

            if self.resize!=(1.0,1.0): #(512, 1024)
                im_width, im_height = img.size
                label_width, label_height = label.size
                img = img.resize((int(im_width * self.resize[1]), int(im_height * self.resize[0])),
                                 resample=PIL.Image.BICUBIC)
                label = label.resize((int(label_width * self.resize[1]), int(label_height * self.resize[0])),
                                     resample=PIL.Image.NEAREST)
            if self.keep_origin: #paste mode
                img_lb_crop = self.random_crop(dict(im=img.copy(), lb=label.copy()))
                img_crop, label_crop = img_lb_crop['im'], img_lb_crop['lb']
                img_origin = self.to_tensor_only(img_crop) #(3,900,900)
                img_origin_norm = self.to_tensor(img_crop) #(3,900,900)
                label_origin = np.array(label_crop).astype(np.int64)[np.newaxis, :] #(1,900,900)
                label_origin = self.convert_labels(label_origin) #(1,900,900)
            #augment the original image
            if self.if_augmented:
                im_lb = dict(im=img, lb=label)
                im_lb = self.trans_train(im_lb)
                img, label = im_lb['im'], im_lb['lb']
            #then normalize the tensor
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            if not self.pseudo_dir:
                label = self.convert_labels(label)
            if self.keep_origin:
                return img, label, os.path.basename(impth).split('.')[0], img_origin, label_origin, img_origin_norm #tensor, np.array
            else:
                return img, label, os.path.basename(impth).split('.')[0]
        #unsup single
        elif self.mode=='unsup_single':
            fn = self.imnames[idx]
            impth = self.imgs[fn]
            img = Image.open(impth)
            if self.resize!=(1.0,1.0):
                im_width, im_height = img.size
                img=img.resize((int(im_width * self.resize[1]), int(im_height * self.resize[0])), 
                                resample=PIL.Image.BICUBIC)
            if self.keep_origin:
                img_origin = self.to_tensor_only(img.copy())
                img_origin_norm = self.to_tensor(img.copy())
            if self.if_augmented:
                img=self.trans_unsup_image(img) #flip scale crop
            img_tensor = self.to_tensor(img)
            if self.keep_origin:
                return img_tensor, os.path.basename(impth).split('.')[0], img_origin, img_origin_norm
            else:
                return img_tensor, os.path.basename(impth).split('.')[0]
        elif self.mode=='test':
            fn = self.imnames[idx]
            impth = self.imgs[fn]
            img = Image.open(impth)
            img = self.to_tensor(img)
            return img, os.path.basename(impth).split('.')[0]
        #unsup
        else:
            fn = self.imnames[idx]
            impth = self.imgs[fn]
            img = Image.open(impth)
            if self.resize != (1.0, 1.0):
                im_width, im_height = img.size
                img = img.resize((int(im_width * self.resize[1]), int(im_height * self.resize[0])),
                                 resample=PIL.Image.BICUBIC)
            img = self.trans_unsup_image(img) #weak transform, flip, scale, crop
            if self.keep_origin:
                img_origin = self.to_tensor_only(img.copy())
                img_origin_norm = self.to_tensor(img.copy())
            img_tensor = self.to_tensor(img)
            img_trans = self.trans_unsup_color(img) #texture augment, color jitter, shadow, noise, blur, style
            img_trans_tensor = self.to_tensor(img_trans) #to tensor and normalize
            img_trans_tensor, affine1_to_2, _ = random_affine(img_trans_tensor, **self.affine_kwargs) #structure transform
            if self.keep_origin:
                return img_tensor, img_trans_tensor, affine1_to_2, os.path.basename(impth).split('.')[0], img_origin, img_origin_norm #tensor, before normalize
            else:
                return img_tensor, img_trans_tensor, affine1_to_2, os.path.basename(impth).split('.')[0]


    def __len__(self):
        return self.len

    def convert_labels(self, label):
        label_copy = 255 * np.ones(label.shape, dtype=np.int64)
        for k, v in self.lb_map.items():
            label_copy[label == k] = v
        return label_copy

if __name__ == "__main__":
    affine_kwargs = {"rot": 20.0,
                     "shear": 5.0,
                     "scale": 0.8,
                     }
    trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5)),
            RandomCrop((512,512)),
            Shadow(shadow=(0.01, 0.3),
                   shadow_file=os.path.join(dir_path, "shadow_pattern.jpg"),
                   shadow_crop_range=(0.02, 0.5)),
            Noise(noise=4),
            Blur(blur=0.2),
        ])
    trans_unsup_image = Compose([
            HorizontalFlip_image(),
            RandomScale_image((0.75, 1.0, 1.25, 1.5)),
            RandomCrop_image((512,512))
        ])
    texture_trans=Compose([
        ColorJitter_image(
            brightness=[0.4, 1.2],
            contrast=[0.4, 2],
            saturation=[0.5, 2],
            sharpeness=[0.4, 2]),
        Shadow_image(shadow=(0.01, 0.3),
                     shadow_file=os.path.join(dir_path, "shadow_pattern.jpg"),
                     shadow_crop_range=(0.02, 0.5)),
        Noise_image(noise=[3, 5]),
        Blur_image(blur=0.2),
        Style_image(p=0.3)])

    impth="/media/ywh/ubuntu/Dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_001236_leftImg8bit.png"
    img = Image.open(impth)
    img = img.resize((1024,512),
                     resample=PIL.Image.BICUBIC)
    img.show()
    to_tensor=transforms.ToTensor()
    to_pil=transforms.ToPILImage()
    to_tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    img_texture=texture_trans(img)
    img_texture.show()








