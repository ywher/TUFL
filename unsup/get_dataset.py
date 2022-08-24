import sys, os
import torch
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from datasets.cityscapes import CityScapes
from datasets.crosscity import CrossCity

def get_source_dataset(args):
    if 'CityScapes' in args.dataset: #'CityScapes', 'CityScapes_Rome', 'CityScapes_Rio', 'CityScapes_Tokyo', 'CityScapes_Taipei'
        n_classes = args.n_classes
        mode = 'train'
        if args.dataset in ['CityScapes_Rome', 'CityScapes_Rio', 'CityScapes_Tokyo', 'CityScapes_Taipei']:
            mode += args.dataset[args.dataset.find('_'):] #add the city name
        dataroot = os.path.join(root_folder, 'data', 'cityscapes') #need to change
        crop_size = (1024,1024) if args.supervision_mode == 'sup' else (640,1280)
        trainset = CityScapes(dataroot, cropsize=crop_size, mode=mode, n_class=n_classes, data_ratio=args.sup_ratio, keep_origin=(args.paste_mode != 'None'))
    sampler_trainset = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.n_img_per_gpu_train, 
        shuffle=False, num_workers=args.n_workers, pin_memory=True, sampler=sampler_trainset, drop_last=True)
    return n_classes, crop_size, dataroot, trainset, sampler_trainset, trainloader

def get_taret_dataset(args, crop_size, n_classes):
    if args.dataset in ['CityScapes', 'CityScapes_Rome', 'CityScapes_Rio', 'CityScapes_Tokyo', 'CityScapes_Taipei']:
        unsuproot = os.path.join(root_folder, 'data', 'NTHU')
        if args.paste_mode != 'None':
            unsupset = CrossCity(unsuproot, cropsize=crop_size, mode=args.supervision_mode, city=args.target_city, n_class=n_classes, keep_origin=True, pseudo_dir=args.pseudo_save_dir)
        else:
            unsupset = CrossCity(unsuproot, cropsize=crop_size, mode=args.supervision_mode, city=args.target_city, n_class=n_classes)
    sampler_unsupset = torch.utils.data.distributed.DistributedSampler(unsupset)
    unsuploader = torch.utils.data.DataLoader(unsupset, batch_size=args.n_img_per_gpu_unsup, 
        shuffle=False, num_workers=args.n_workers, pin_memory=True, sampler=sampler_unsupset, drop_last=True)
    return unsupset, sampler_unsupset, unsuploader