import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc

import torch
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab
import glob
from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from PIL import Image
from torchvision import transforms
import json
import imageio

from datasets.transform import *
from datasets.transform_image import *

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)



def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_ammend(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

###use
def strongTransform_class_mix(img_temp, image_src, image_tar, label_temp, label_src, label_tar, one_mask, cls_mixer, cls_list, strong_parameters, mixWeight=1.0):
    """
    Args:
        img_temp, label_temp: template image and label to paste
        image_src, image_tar, label_src, label_tar: image and label from source and target
        one_mask: one-mask extracted from img_temp, element value is 1 or 0
        cls_mixer(obj:rand_mixer): to mix img_temp and label_temp with long tail classes
        cls_list(list): long tail classes to select
        strong_parameters: data augmentation method
        mixWeight(float): to control the pixel weight of img_temp
    """
    img_temp, label_temp, mixed_mask = cls_mixer.mix(img_temp, label_temp, one_mask, cls_list) #(1,3,H,W), (1,H,W), (1,H,W)
    mask_img = mixed_mask * mixWeight
    mask_lbl = mixed_mask

    image_src_mix_lt, _ = transformsgpu.oneMix(mask_img, data=torch.cat((img_temp, image_src)))  # image_src with long tail mixed
    image_tar_mix_lt, _ = transformsgpu.oneMix(mask_img, data=torch.cat((img_temp, image_tar)))

    _, label_src_mix_lt = transformsgpu.oneMix(mask_lbl.long(), target=torch.cat((label_temp, label_src)))  # label_src with long tail mixd
    _, label_tar_mix_lt = transformsgpu.oneMix(mask_lbl.long(), target=torch.cat((label_temp, label_tar)))

    # out_img_src_mix, out_lbl_src_mix = strongTransform_ammend(strong_parameters, data=image_src_mix_lt, target=label_src_mix_lt)
    # out_img_tar_mix, out_lbl_tar_mix = strongTransform_ammend(strong_parameters, data=image_tar_mix_lt, target=label_tar_mix_lt)
    # return out_img_src_mix, out_lbl_src_mix, out_img_tar_mix, out_lbl_tar_mix, mask_img
    
    return image_src_mix_lt, label_src_mix_lt, image_tar_mix_lt, label_tar_mix_lt, mask_img

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def loss_calc(pred, label, ignore_label, gpus):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, learning_rate, num_iterations, lr_power):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model, num_classes, gpus):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration, gpus):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages', str(epoch)+ id + '.png'))

def save_checkpoint(miou,iteration, model, optimizer, config, ema_model, checkpoint_dir, train_unlabeled, save_best=False, overwrite=True, gpus=1):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filelist = glob.glob(os.path.join(checkpoint_dir,'*.pth'))
        if filelist:
            os.remove(filelist[0])
        filename = os.path.join(checkpoint_dir, f'{miou}best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

###use
class rand_mixer():
    def __init__(self, root, dataset, cropsize):
        #root: str, path to the dataset
        #dataset: str, type of dataset
        #cropsize: tuple, (width, height)
        if dataset == "gta":
            jpath = './datasets/gta5_ids2path.json'
            self.resize = (1.0, 1.0)
            self.input_size = cropsize
            self.data_aug = Compose([RandomCrop(cropsize)])
            self.class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        elif dataset == "synthia":
            jpath = './datasets/synthia_ids2path.json'
            self.resize = (1.0, 1.0)
            self.input_size = cropsize
            self.data_aug = Compose([RandomCrop(cropsize)])
            self.class_map = {1: 9, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8,
                                7: 5, 8: 12, 9: 7, 10: 10, 11: 15, 12: 14, 15: 6,
                                17: 11, 19: 13, 21: 3}
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
        self.root = root
        self.dataset = dataset

        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)

    def mix(self, in_img, in_lbl, one_mask, classes, weight=1.0):
        #in_img: (3,H,W)
        #in_lbl: (1,H,W)
        #one_mask: (1,H,W)
        #classes: list of long_tail classes [class1, class2]
        #weight: float
        in_img = in_img.unsqueeze(0) #(1,3,H,W)
        in_lbl = in_lbl.unsqueeze(0) #(1,1,H,W)
        one_mask = one_mask.squeeze(0) #(H,W)
        # added_mask = one_mask.cpu() * 0
        for i in classes:
            #paste one class to the in_image and in_lb once a time
            if self.dataset == "gta":
                loopi = 0
                while(True):
                    loopi = loopi + 1
                    if(loopi>1000):
                        print("Warning: No long tail image match")
                        break
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    img_path = os.path.join(self.root, "images/%s" % name[0])
                    label_path = os.path.join(self.root, "labels/%s" % name[0])
                    img = Image.open(img_path) #(H,W,3)
                    lbl = Image.open(label_path) #(H,W)
                    if self.resize != (1.0,1.0):
                        img = img.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.BICUBIC)
                        lbl = lbl.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.NEAREST)
                    im_lb = dict(im=img, lb=lbl)
                    im_lb = self.data_aug(im_lb)
                    img, lbl = im_lb['im'], im_lb['lb'] #random crop to cropsize
                    
                    # img = np.array(img, dtype=np.uint8)
                    # lbl = np.array(lbl, dtype=np.uint8)
                    # img, lbl = self.data_aug(img, lbl) # random crop to input_size
                    img = np.array(img, dtype=np.float32)
                    lbl = np.array(lbl, dtype=np.float32)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.float32) #(H,W)
                    for k, v in self.class_map.items(): #translate label map
                        label_copy[lbl == k] = v
                    if i in label_copy: #long tail calss may be cropped from the image
                        lbl = label_copy.copy()
                        img = img[:, :, ::-1].copy()  # change to BGR
                        #img -= IMG_MEAN #may not need to minus it, we keep origin img and lb and use normalize
                        img = img.transpose((2, 0, 1)) #(H,W,C)->(C,H,W)
                        break

            if self.dataset == "synthia":
                loopi = 0
                while(True):
                    loopi = loopi + 1
                    if(loopi>1000):
                        print("Warning: No long tail image match")
                        break
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    img_path = os.path.join(self.root, "RGB/%s" % name[0])
                    label_path = os.path.join(self.root, "GT/LABELS/%s" % name[0])
                    img = Image.open(img_path).convert('RGB')
                    lbl = np.asarray(imageio.imread(label_path, format='PNG-FI'))[:,:,0]  # uint16
                    lbl = Image.fromarray(lbl)
                    if self.resize != (1.0,1.0):
                        img = img.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.BICUBIC)
                        lbl = lbl.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.NEAREST)
                    im_lb = dict(im=img, lb=lbl)
                    im_lb = self.data_aug(im_lb)
                    img, lbl = im_lb['im'], im_lb['lb'] #random crop to cropsize
                    img = np.array(img, dtype=np.float32)
                    lbl = np.array(lbl, dtype=np.float32)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
                    for k, v in self.class_map.items():
                        label_copy[lbl == k] = v
                    if i in label_copy: #(B)
                        lbl = label_copy.copy()
                        img = img[:, :, ::-1].copy()  # change to BGR
                        # img -= IMG_MEAN
                        img = img.transpose((2, 0, 1))
                        break

            img = torch.Tensor(img).cuda() #(3,H,W), long_tail class image
            lbl = torch.Tensor(lbl).cuda() #(H,W)
            class_i = torch.Tensor([i]).type(torch.int64).cuda() 
            MixMask = transformmasks.generate_class_mask(lbl, class_i)#(H,W),(N)->(1,H,W)

            mixdata = torch.cat((img.unsqueeze(0), in_img)) #(1,3,H,W), (1,3,H,W)
            mixtarget = torch.cat((lbl.unsqueeze(0), in_lbl)) #(1,H,W), (1,H,W)
            in_img, _ = transformsgpu.oneMix(MixMask * weight, data=mixdata)
            _, in_lbl = transformsgpu.oneMix(MixMask, target=mixtarget)

            one_mask[MixMask == 1] = 1
            # added_mask[MixMask.cpu() == 1] = 1
        # print('long tail class {} percent:{:.2f}%'.format(i, 100*added_mask.sum()/(input_size[0]*input_size[1])))
        return in_img, in_lbl, one_mask.unsqueeze(0) #(1,3,H,W), (1,H,W), (1,H,W)
