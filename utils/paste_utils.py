import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from utils import transformmasks
from utils import transformsgpu
import torch
from datasets.transform import *
from datasets.transform_image import *
import json
import os
from PIL import Image
from torchvision import transforms
from scipy.ndimage import distance_transform_edt

def strongTransform_class_mix(img_temp, image_src, image_tar, label_temp, label_src, label_tar, one_mask, cls_mixer, cls_list, strong_parameters=None, mixWeight=1.0):
    """
    Args:
        img_temp, label_temp: template image and label to paste (1,3,H,W), (1,H,W)
        image_src, image_tar, label_src, label_tar: image and label from source and target (1,3,H,W), (1,H,W)
        one_mask: one-mask extracted from img_temp, element value is 1 or 0 (1,H,W)
        cls_mixer(obj:rand_mixer): to mix img_temp and label_temp with long tail classes
        cls_list(list): long tail classes to select
        strong_parameters: data augmentation method NOne
        mixWeight(float): to control the pixel weight of img_temp 0.8
    """
    img_temp, label_temp, mixed_mask = cls_mixer.mix(img_temp, label_temp, one_mask, cls_list) #(1,3,H,W), (1,H,W), (1,H,W)
    mask_img = mixed_mask * mixWeight
    mask_lbl = mixed_mask

    image_src_mix_lt, _ = transformsgpu.oneMix(mask_img, data=torch.cat((img_temp, image_src)))  # image_src with long tail mixed
    image_tar_mix_lt, _ = transformsgpu.oneMix(mask_img, data=torch.cat((img_temp, image_tar)))

    _, label_src_mix_lt = transformsgpu.oneMix(mask_lbl.long(), target=torch.cat((label_temp, label_src)))  # label_src with long tail mixd
    _, label_tar_mix_lt = transformsgpu.oneMix(mask_lbl.long(), target=torch.cat((label_temp, label_tar)))
    
    return image_src_mix_lt, label_src_mix_lt, image_tar_mix_lt, label_tar_mix_lt, mask_img


class rand_mixer():
    def __init__(self, root, dataset, cropsize):
        #root: str, path to the dataset
        #dataset: str, type of dataset
        #cropsize: tuple, (width, height)
        if dataset == "gta":
            jpath = './datasets/gta5_ids2path.json'
            self.resize = (1.0, 1.0)
            self.input_size = cropsize
            self.data_aug = Compose([RandomCrop((cropsize[1],cropsize[0]))])
            self.class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
            self.class_relation = {
                0:[0,9,12,13,14,15,16,17,18],
                1:[1,9,11],
                2:[],
                3:[3,4],
                4:[3,4],
                5:[5,6,7],
                6:[5,6,7],
                7:[5,6,7],
                8:[8,9],
                9:[8,9],
                10:[],
                11:[],
                12:[12,17,18],
                13:[],
                14:[],
                15:[],
                16:[],
                17:[12,17,18],
                18:[12,17,18],
                255:[]
                }
        elif dataset == "synthia":
            jpath = './datasets/synthia_ids2path.json'
            self.resize = (1.0, 1.0)
            self.input_size = cropsize
            self.data_aug = Compose([RandomCrop((cropsize[1],cropsize[0]))])
            self.class_map = {1: 9, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8,
                                7: 5, 8: 12, 9: 7, 10: 10, 11: 15, 12: 14, 15: 6,
                                17: 11, 19: 13, 21: 3}
            self.class_relation={
                0:[0,10,11,12,13,14,15],
                1:[1,10,11,14,15],
                2:[2,5,6,7],
                3:[],
                4:[],
                5:[5,6,7],
                6:[5,6,7],
                7:[5,6,7],
                8:[],
                9:[],
                10:[],
                11:[11,14,15],
                12:[],
                13:[],
                14:[11,14,15],
                15:[11,14,15],
                255:[]
            }
        elif dataset == "cityscapes":
            jpath = './datasets/cityscapes_ids2path.json'
            self.resize = (1.0, 1.0)
            self.input_size = cropsize
            self.data_aug = Compose([RandomCrop((cropsize[1],cropsize[0]))])
            self.class_map = {7:0,8:1,11:2,19:3,20:4,21:5,23:6,24:7,25:8,26:9,28:10,32:11,33:12}
            self.class_relation={
                0:[0,10,11,12],
                1:[1,10,11],
                2:[2,5,6,7],
                3:[],
                4:[],
                5:[5,6,7],
                6:[5,6,7],
                7:[5,6,7],
                8:[],
                9:[],
                10:[],
                11:[],
                12:[],
                255:[]
            }
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
        self.root = root
        self.dataset = dataset
        self.to_tensor = transforms.ToTensor()

        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)

    def mix(self, in_img, in_lbl, one_mask, classes, weight=1.0):
        #in_img: (1,3,H,W)
        #in_lbl: (1,H,W)
        #one_mask: (1,H,W)
        #classes: list of long_tail classes [class1, class2]
        #weight: float
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
                    lbl = np.array(lbl).astype(np.int64) #(1024,1024)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.int64) #(H,W)
                    for k, v in self.class_map.items(): #translate label map
                        label_copy[lbl == k] = v
                    if i in label_copy: #long tail calss may be cropped from the image
                        lbl = label_copy.copy()
                        # img = img[:, :, ::-1].copy()  # change to BGR
                        # #img -= IMG_MEAN #may not need to minus it, we keep origin img and lb and use normalize
                        # img = img.transpose((2, 0, 1)) #(H,W,C)->(C,H,W)
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
                    img = Image.open(img_path) #(760,1280,3)
                    lbl = Image.open(label_path) #(760,1280)
                    if self.resize != (1.0,1.0):
                        img = img.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.BICUBIC)
                        lbl = lbl.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.NEAREST)
                    im_lb = dict(im=img, lb=lbl)
                    im_lb = self.data_aug(im_lb)
                    img, lbl = im_lb['im'], im_lb['lb'] #random crop to cropsize, (760,1280,3) (760,1280)
                    lbl = np.array(lbl).astype(np.int64) #(760,1280)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.int64)
                    for k, v in self.class_map.items():
                        label_copy[lbl == k] = v
                    if i in label_copy: #(B)
                        lbl = label_copy.copy()
                        # img = img[:, :, ::-1].copy()  # change to BGR
                        # img -= IMG_MEAN
                        # img = img.transpose((2, 0, 1))
                        break

            if self.dataset == "cityscapes":
                loopi = 0
                while(True):
                    loopi = loopi + 1
                    if(loopi>1000):
                        print("Warning: No long tail image match")
                        break
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    img_path = os.path.join(self.root, "leftImg8bit", "train", name[0])
                    label_path = os.path.join(self.root, "gtFine", "train", name[0])
                    img = Image.open(img_path) #(760,1280,3)
                    lbl = Image.open(label_path) #(760,1280)
                    if self.resize != (1.0,1.0):
                        img = img.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.BICUBIC)
                        lbl = lbl.resize((int(self.resize[1]), int(self.resize[0])), resample=Image.NEAREST)
                    im_lb = dict(im=img, lb=lbl)
                    im_lb = self.data_aug(im_lb)
                    img, lbl = im_lb['im'], im_lb['lb'] #random crop to cropsize, (760,1280,3) (760,1280)
                    lbl = np.array(lbl).astype(np.int64) #(760,1280)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.int64)
                    for k, v in self.class_map.items():
                        label_copy[lbl == k] = v
                    if i in label_copy: #(B)
                        lbl = label_copy.copy()
                        break
            
            img = self.to_tensor(img).cuda() #(3,H,W), long_tail class imagem float32
            lbl = torch.Tensor(lbl).cuda() #(H,W), float32
            all_classes = list((torch.unique(lbl)).cpu().numpy())
            update_class = [i]
            if len(self.class_relation[i]) != 0:
                for related_class in self.class_relation[i]:
                    if (related_class in all_classes) and (related_class not in update_class):
                        update_class.append(related_class)

            class_i = torch.Tensor(update_class).type(torch.int64).cuda() 
            MixMask = transformmasks.generate_class_mask(lbl.long(), class_i.long()).unsqueeze(0).cuda().float()#(H,W),(N)->(1,H,W), float32
            mixdata = torch.cat((img.unsqueeze(0), in_img)) #(1,3,H,W), (1,3,H,W)
            mixtarget = torch.cat((lbl.unsqueeze(0).type(torch.int64), in_lbl)) #(1,H,W), (1,H,W)
            
            in_img, _ = transformsgpu.oneMix(MixMask * weight, data=mixdata)
            _, in_lbl = transformsgpu.oneMix(MixMask.long(), target=mixtarget)

            one_mask[MixMask == 1] = 1
        return in_img, in_lbl, one_mask #(1,3,H,W), (1,H,W), (1,H,W)


class pick_mix_class():
    def __init__(self, dataset):
        if dataset == 'synthia':
            self.class_relation = {
                0:[10,11,12,13,14,15],
                1:[10,11,14,15],
                2:[5,6,7],
                3:[],
                4:[],
                5:[6,7],
                6:[5,7],
                7:[5,6],
                8:[],
                9:[],
                10:[],
                11:[14,15],
                12:[],
                13:[],
                14:[11,15],
                15:[11,14],
                255:[]
            }
        elif dataset == 'gta':
            self.class_relation = {
                0:[9,12,13,14,15,16,17,18],
                1:[9,11],
                2:[],
                3:[4],
                4:[3],
                5:[6,7],
                6:[5,7],
                7:[5,6],
                8:[9],
                9:[8],
                10:[],
                11:[],
                12:[17,18],
                13:[],
                14:[],
                15:[],
                16:[],
                17:[12,18],
                18:[12,17],
                255:[]
            }
        elif dataset == 'cityscapes':
            self.class_relation = {
                0:[10,11,12],
                1:[10,11],
                2:[5,6,7],
                3:[],
                4:[],
                5:[6,7],
                6:[5,7],
                7:[5,6],
                8:[],
                9:[],
                10:[],
                11:[],
                12:[],
                255:[]
            }
    def get_pick_classes(self, all_classes, pick_classes):
        #all_classes: list [class1, class2, ...], all classes in the image
        #pick_classes: list [class1, class2, ...], half of classes in the image

        #remove unlabeled class
        if 255 in pick_classes:
            pick_classes.remove(255)
        all_numbers = len(all_classes)
        update_classes = []
        for cls in pick_classes:
            related_classes = self.class_relation[cls]
            if cls not in update_classes:
                update_classes.append(cls)
            if len(related_classes) != 0:
                for related_class in related_classes:
                    if (related_class in all_classes) and (related_class not in update_classes):
                        update_classes.append(related_class)
            if len(update_classes) >= (all_numbers//2):
                break
        return update_classes

def compute_edts_forPenalizedLoss(GT, d):
    """
    GT.shape = (batch_size, H, W), (1,760,1280) torch.tensor.float().cuda()
    only for binary segmentation
    """
    GT = GT.cpu().numpy() #(1,760,1280)
    GT = GT.astype(np.bool)
    # GT = np.squeeze(GT)
    res = np.zeros(GT.shape) #(1,760,1280)
    for i in range(GT.shape[0]):
        posmask = GT[i] #(760,1280)
        negmask = ~posmask #
        pos_edt = distance_transform_edt(posmask) #eduli distance
        pos_edt = (np.max(pos_edt) - pos_edt) * posmask * (pos_edt<d)
        neg_edt = distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt) - neg_edt) * negmask * (neg_edt<d)
        res[i] = pos_edt / np.max(pos_edt) + neg_edt / np.max(neg_edt)
    return res


def update_ema_variables(ema_model, model, alpha_teacher, iteration, gpus):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
            #this line should be checked for correctness, .data[:], [:].data[:], [:].data[:]
            # print(ema_param[:].data[:])
            # print(param[:].data[:])
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

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

def save_tensor_image(tensor_img, file_path):
    #tensor_img (1,C,H,W)
    from torchvision import utils as vutils
    assert len(tensor_img.shape)==4 and tensor_img.shape[0] == 1
    tensor_img = tensor_img.clone().detach()
    tensor_img = tensor_img.to(torch.device('cpu'))
    vutils.save_image(tensor_img, file_path)

def colorize_mask(mask):
    #mask tensor (1,1,H,W)
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

if __name__ == '__main__':
    # pick_class = pick_mix_class('synthia')
    # all_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # pick_classes = [5,8,11]
    # update_class = pick_class.get_pick_classes(all_classes=all_classes, pick_classes=pick_classes)
    # print('update_class: ', update_class)
    a = [[0,1,1,1,1,1,1],[0,0,1,1,1,1,1],[0,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,1,1,1,1,0,0,],[0,1,1,0,0,0,1],[0,1,1,1,0,0,1]]
    a = torch.tensor(a).unsqueeze(0).cuda()
    d = 3
    res = compute_edts_forPenalizedLoss(a,d)