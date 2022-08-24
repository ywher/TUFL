import torch.nn as nn
import torchvision.transforms as transforms
import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from utils.paste_utils import *
def get_func(args, dataroot, crop_size):
    to_normalize, to_pil, gaussian_blur, max_pool, lt_cls_mixer, class_to_select, pick_class_function = None, None, None, None, None, None, None
    if args.paste_mode != 'None':
        to_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        to_pil = transforms.ToPILImage()
    if args.mixed_gaussian_kernel > 0:
        gaussian_blur = transforms.GaussianBlur(kernel_size=args.mixed_gaussian_kernel)
    if args.mixed_boundary > 0:
        max_pool = nn.MaxPool2d(kernel_size=args.mixed_boundary, stride=1, padding=(args.mixed_boundary//2))
    #long tail class paste
    if args.paste_mode == 'Single' and args.long_tail:
        if 'GTA5' in args.dataset:
            lt_cls_mixer = rand_mixer(dataroot, "gta", cropsize=crop_size)
            #class_to_select = [12, 15, 16, 17, 18] #rider, bus, train, motor, bike
            #class_to_select = [4, 5, 9, 11, 12, 15, 16, 17] #sidewalk, wall, fence, pole, traffic light, traffic sign, terrain, rider, truck, bus, train, motor, bike
            class_to_select = np.array(range(19))
            pick_class_function = pick_mix_class('gta')
        elif 'Synthia' in args.dataset:
            lt_cls_mixer = rand_mixer(dataroot, "synthia", cropsize=crop_size) #mix long-tail classes with source img
            #class_to_select = [3, 4, 6, 13] #wall, fence, traffic light bus
            # class_to_select = [4, 5, 9, 11, 12, 15, 16, 17] #fence, pole, terrain, person, rider, bus, train, motor
            class_to_select = np.array(range(16))
            pick_class_function = pick_mix_class('synthia')
        elif 'CityScapes' in args.dataset:
            lt_cls_mixer = rand_mixer(dataroot, 'cityscapes', cropsize=crop_size)
            class_to_select = np.array(range(13))
            pick_class_function = pick_mix_class('cityscapes')
    elif args.paste_mode == 'Dual_soft' and args.long_tail:
        if 'GTA5' in args.dataset:
            lt_cls_mixer = rand_mixer(dataroot, "gta", cropsize=crop_size)
            class_to_select = [12, 15, 16, 17, 18] #rider, bus, train, motor, bike
            pick_class_function = pick_mix_class('gta')
        elif 'Synthia' in args.dataset:
            lt_cls_mixer = rand_mixer(dataroot, "synthia", cropsize=crop_size) #mix long-tail classes with source img
            class_to_select = [3, 4, 6, 13] #wall, fence, traffic light bus
            pick_class_function = pick_mix_class('synthia')
        elif 'CityScapes' in args.dataset:
            lt_cls_mixer = rand_mixer(dataroot, 'cityscapes', cropsize=crop_size)
            class_to_select = np.array(range(13))
            pick_class_function = pick_mix_class('cityscapes')

    return to_normalize, to_pil, gaussian_blur, max_pool, lt_cls_mixer, class_to_select, pick_class_function