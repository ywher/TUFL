import torch
import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from utils.logger import iou
from datasets.transform_image import perform_affine_tf
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import time
import datetime
class train_step():
    def __init__(self, args, criteria_p, criteria_unsup, criteria_paste, n_classes):
        super(train_step, self).__init__()
        self.img_count = 0
        self.start_time = time.time()
        self.glob_st = time.time()
        self.to_pil = transforms.ToPILImage()
        self.args = args #cofig parameters
        self.criteria_p = criteria_p #supervised loss
        self.criteria_unsup = criteria_unsup #unsupervised loss
        self.criteria_paste = criteria_paste #mixed loss
        self.n_classes = n_classes
        self.soft_weight = args.soft_weight
        self.mix_img_save_pth = os.path.join(root_folder, 'outputs', 'images', 'mix')
        self.total_loss_list = [] #total loss
        self.loss_seg_list = [] #supervised loss
        self.loss_unsup_list = [] #unsupvised loss
        self.loss_mix_list = [] #mix supervised loss
        if self.args.paste_mode == 'Dual_soft':
            self.loss_mix_src_list = []
            self.loss_mix_tar_list = []
        self.iou_train_list = [] #train batch iou
        if self.args.unsup_loss == 'adapt_focal': #adapt focal ratio
            self.adapt_ratio_list, self.adapt_ratio16_list, self.adapt_ratio32_list = [],[],[]
            self.solid_ratio_list, self.solid_ratio16_list, self.solid_ratio32_list = [],[],[]


    def sup_step(self, im, lb, optim, net):
        im_all = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb,1)

        optim.zero_grad()
        if self.args.segmentation_model == 'BiSeNet':
            out_all, out16_all, out32_all, _ = net(im_all)
            out = out_all[:self.args.n_img_per_gpu_train]
            out16 = out16_all[:self.args.n_img_per_gpu_train]
            out32 = out32_all[:self.args.n_img_per_gpu_train]
            loss_sup = (self.criteria_p(out, lb) + self.criteria_p(out16, lb) + self.criteria_p(out32, lb))/3
            loss = loss_sup
        mean_iou_train = iou(out, lb, self.n_classes)
        self.iou_train_list.append(mean_iou_train)
        self.loss_seg_list.append(loss_sup.item())
        self.total_loss_list.append(loss.item())
        return loss, optim, net

    def unsup_single_step(self, im, lb, im_unsup, optim, net):
        #merge the surpvised and unsupervised image
        im_all = torch.cat([im, im_unsup]).cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out_all, out16_all, out32_all,_ = net(im_all)
        
        #source supervised part
        out = out_all[:self.args.n_img_per_gpu_train]
        out16 = out16_all[:self.args.n_img_per_gpu_train]
        out32 = out32_all[:self.args.n_img_per_gpu_train]

        loss_sup = (self.criteria_p(out, lb) + self.criteria_p(out16, lb) + self.criteria_p(out32, lb))/3
        mean_iou_train = iou(out, lb, self.n_classes)
        self.iou_train_list.append(mean_iou_train)
        #validation part

        # semi-supervised part
        out_unsup = out_all[self.args.n_img_per_gpu_train:2*self.args.n_img_per_gpu_train]
        out_unsup16 = out16_all[self.args.n_img_per_gpu_train:2*self.args.n_img_per_gpu_train]
        out_unsup32 = out32_all[self.args.n_img_per_gpu_train:2*self.args.n_img_per_gpu_train]
        loss_unsup_p = self.criteria_unsup(out_unsup)
        loss_unsup_16 = self.criteria_unsup(out_unsup16)
        loss_unsup_32 = self.criteria_unsup(out_unsup32)

        loss_unsup = (loss_unsup_p + loss_unsup_16 + loss_unsup_32)/3
        loss = loss_sup + self.args.unsup_coeff * loss_unsup

        self.loss_seg_list.append(loss_sup.item())
        self.loss_unsup_list.append(loss_unsup.item())
        self.total_loss_list.append(loss.item())
        return loss, optim, net

    def unsup_step(self, im, lb, im_unsup, im_unsup_aug, all_affine1_to_2, names_unsup, it, \
        mix_img, mix_lb, source_mix_img, source_mix_lbl, lb_origin, target_mix_img, target_mix_lbl, targets_u_w, pixelWiseWeight, optim, net, pseudo_net):
        if self.args.paste_mode == 'Single':
            im_all = torch.cat([im, im_unsup, im_unsup_aug, mix_img]).cuda()
        elif self.args.paste_mode == 'Dual_soft':
            im_all = torch.cat([im, im_unsup, im_unsup_aug, source_mix_img, target_mix_img]).cuda()
        else:
            im_all = torch.cat([im, im_unsup, im_unsup_aug]).cuda()
        lb = lb.cuda() #(1,1,760,1280)
        lb = torch.squeeze(lb, 1) #dimension reduction (N,H,W)

        optim.zero_grad()
        if self.args.sup_mode == 'deep':
            out_all, out16_all, out32_all,_ = net(im_all)
        else:
            out_all, _, _, _ = net(im_all)

        #source supervised part
        out = out_all[:self.args.n_img_per_gpu_train] #(1,C,H,W)
        if self.args.sup_mode == 'deep':
            out16 = out16_all[:self.args.n_img_per_gpu_train]
            out32 = out32_all[:self.args.n_img_per_gpu_train]
        if self.args.sup_mode == 'deep':
            loss_sup = (self.criteria_p(out, lb) + self.criteria_p(out16, lb) + self.criteria_p(out32, lb))/3
        elif self.args.sup_mode == 'last':
            loss_sup = self.criteria_p(out, lb)
        mean_iou_train = iou(out, lb, self.n_classes)
        self.iou_train_list.append(mean_iou_train)

        # unsupervised part
        preds_unsup = out_all[self.args.n_img_per_gpu_train:3*self.args.n_img_per_gpu_train]
        out_unsup, out_unsup_after = torch.chunk(preds_unsup, 2)
        if self.args.sup_mode == 'deep':
            preds_unsup16 = out16_all[self.args.n_img_per_gpu_train:3*self.args.n_img_per_gpu_train]
            out_unsup16, out_unsup_after16 = torch.chunk(preds_unsup16, 2)
            preds_unsup32 = out32_all[self.args.n_img_per_gpu_train:3*self.args.n_img_per_gpu_train]
            out_unsup32, out_unsup_after32 = torch.chunk(preds_unsup32, 2)

        all_affine1_to_2 = all_affine1_to_2.cuda()
        assert (not all_affine1_to_2.requires_grad)

        out_unsup = perform_affine_tf(out_unsup, all_affine1_to_2)
        if self.args.sup_mode == 'deep':
            out_unsup16 = perform_affine_tf(out_unsup16, all_affine1_to_2)
            out_unsup32 = perform_affine_tf(out_unsup32, all_affine1_to_2)
        if self.args.unsup_loss == 'adapt_focal':
            loss_unsup_p, adapt_ratio, solid_ratio  = self.criteria_unsup(out_unsup, out_unsup_after, names_unsup, it)
            self.adapt_ratio_list.append(adapt_ratio)
            self.solid_ratio_list.append(solid_ratio)
            if self.args.sup_mode == 'deep':
                loss_unsup_16, adapt_ratio16, solid_ratio16 = self.criteria_unsup(out_unsup16, out_unsup_after16, None, it)
                loss_unsup_32, adapt_ratio32, solid_ratio32 = self.criteria_unsup(out_unsup32, out_unsup_after32, None, it)
                self.adapt_ratio16_list.append(adapt_ratio16)
                self.adapt_ratio32_list.append(adapt_ratio32)
                self.solid_ratio16_list.append(solid_ratio16)
                self.solid_ratio32_list.append(solid_ratio32)
        else:
            loss_unsup_p  = self.criteria_unsup(out_unsup,out_unsup_after)
            if self.args.sup_mode == 'deep':
                loss_unsup_16 = self.criteria_unsup(out_unsup16, out_unsup_after16)
                loss_unsup_32 = self.criteria_unsup(out_unsup32, out_unsup_after32)
        if self.args.sup_mode == 'deep':
            loss_unsup=(loss_unsup_p+loss_unsup_16+loss_unsup_32)/3
        else:
            loss_unsup = loss_unsup_p
        
        # mixed part
        if self.args.paste_mode == 'Single':
            mix_pred = out_all[3*self.args.n_img_per_gpu_train:]
            loss_mix = self.criteria_paste(mix_pred, mix_lb, pixelWiseWeight) #hard label. one from pseudo, one from gt
            self.loss_mix_list.append(loss_mix.item())
        elif self.args.paste_mode == 'Dual_soft':
            src_mix_pred = out_all[3*self.args.n_img_per_gpu_train:4*self.args.n_img_per_gpu_train]
            tar_mix_pred = out_all[4*self.args.n_img_per_gpu_train:]
            loss_mix_source = self.criteria_paste(src_mix_pred, source_mix_lbl, torch.ones(source_mix_lbl.shape).cuda()*self.soft_weight) \
                    + self.criteria_paste(src_mix_pred, lb_origin, torch.ones(lb_origin.shape).cuda()*(1-self.soft_weight))
            loss_mix_target = self.criteria_paste(tar_mix_pred, target_mix_lbl, pixelWiseWeight*self.soft_weight) \
                    + self.criteria_paste(tar_mix_pred, targets_u_w, pixelWiseWeight*(1-self.soft_weight))
            loss_mix = self.args.src_mix_weight * loss_mix_source + self.args.tar_mix_weight * loss_mix_target
            self.loss_mix_list.append(loss_mix.item())
            self.loss_mix_src_list.append(loss_mix_source.item())
            self.loss_mix_tar_list.append(loss_mix_target.item())
        self.img_count += 1
        
        self.loss_seg_list.append(loss_sup.item())       
        self.loss_unsup_list.append(loss_unsup.item())

        loss = loss_sup + self.args.unsup_coeff * loss_unsup if not (torch.isnan(loss_unsup).any()) else loss_sup
        if self.args.paste_mode != 'None':
            loss = loss + self.args.mixed_weight * loss_mix if not (torch.isnan(loss_mix).any()) else loss
        self.total_loss_list.append(loss.item())
        return loss, optim, net, pseudo_net

    def msg_iter(self, writer, logger, it, optim):
        iou_train_avg = sum(self.iou_train_list) / len(self.iou_train_list) if len(self.iou_train_list) != 0 else -1
        loss_seg_avg  = sum(self.loss_seg_list) / len(self.loss_seg_list) if len(self.loss_seg_list) != 0 else -1
        loss_mix_avg = sum(self.loss_mix_list) / len(self.loss_mix_list) if len(self.loss_mix_list) != 0 else -1
        loss_unsup_avg= sum(self.loss_unsup_list) / len(self.loss_unsup_list) if len(self.loss_unsup_list) != 0 else -1
        total_loss_avg = sum(self.total_loss_list) / len(self.total_loss_list) if len(self.total_loss_list)!=0 else -1
        if self.args.paste_mode == 'Dual_soft':
            loss_mix_src_avg = sum(self.loss_mix_src_list) / len(self.loss_mix_src_list) if len(self.loss_mix_src_list) != 0 else -1
            loss_mix_tar_avg = sum(self.loss_mix_tar_list) / len(self.loss_mix_tar_list) if len(self.loss_mix_tar_list) != 0 else -1
        if self.args.unsup_loss == 'adapt_focal':
            adapt_ratio_avg = sum(self.adapt_ratio_list) / len(self.adapt_ratio_list) if len(self.adapt_ratio_list) != 0 else -1
            adapt_ratio16_avg = sum(self.adapt_ratio16_list) / len(self.adapt_ratio16_list) if len(self.adapt_ratio16_list) != 0 else -1
            adapt_ratio32_avg = sum(self.adapt_ratio32_list) / len(self.adapt_ratio32_list) if len(self.adapt_ratio32_list) != 0 else -1
            solid_ratio_avg = sum(self.solid_ratio_list) / len(self.solid_ratio_list) if len(self.solid_ratio_list) != 0 else -1
            solid_ratio16_avg = sum(self.solid_ratio16_list) / len(self.solid_ratio16_list) if len(self.solid_ratio16_list) != 0 else -1
            solid_ratio32_avg = sum(self.solid_ratio32_list) / len(self.solid_ratio32_list) if len(self.solid_ratio32_list) != 0 else -1
        if self.args.tensorboard:
            scalar_info = {
                'total_loss': total_loss_avg,
                'loss_seg': loss_seg_avg,
                'loss_unsup': loss_unsup_avg,
                'loss_mix': loss_mix_avg,
                'iou_train': iou_train_avg,
            }
            if self.args.paste_mode == 'Dual_soft':
                scalar_info.update({'loss_mix_src':loss_mix_src_avg})
                scalar_info.update({'loss_mix_tar':loss_mix_tar_avg})
            if self.args.unsup_loss == 'adapt_focal':
                if self.n_classes == 16:
                    train_id_list = ['road', 'sidew', 'build', 'wall', 'fence', 'pole', 'tligh',
                                'tsign', 'veget', 'sky', 'perso', 'rider', 'car', 'bus',  'motor', 'bike']
                elif self.n_classes == 19:
                    train_id_list=['road','sidew','build','wall','fence','pole','tligh','tsign','veget','terra',
                        'sky','perso','rider','car','truck','bus','train','motor','bike']
                elif self.n_classes == 13:
                    train_id_list = ['road', 'sidew', 'build', 'tligh',
                                'tsign', 'veget', 'sky', 'perso', 'rider', 'car', 'bus',  'motor', 'bike']
                for i in range(self.n_classes):
                    scalar_info.update({train_id_list[i]:self.criteria_unsup.cls_thresh[i]})
                scalar_info.update({'adapt_ratio': adapt_ratio_avg})
                scalar_info.update({'adapt_ratio16': adapt_ratio16_avg})
                scalar_info.update({'adapt_ratio32': adapt_ratio32_avg})
                scalar_info.update({'solid_ratio': solid_ratio_avg})
                scalar_info.update({'solid_ratio16': solid_ratio16_avg})
                scalar_info.update({'solid_ratio32': solid_ratio32_avg})
            for key, val in scalar_info.items():
                writer.add_scalar(key, val, it)

        lr = optim.lr
        ed = time.time()
        t_intv, glob_t_intv = ed - self.start_time, ed - self.glob_st
        eta = int((self.args.max_iter - it) * (glob_t_intv / it))
        eta = str(datetime.timedelta(seconds=eta))
        msg = ', '.join([
                'it: {it}/{max_it}',
                'eta: {eta}',
                'time: {time:.4f}',
                'lr: {lr:.10f}',
                'loss: {total_loss:.4f}',
                'sup-loss {sup_loss:.4f}',
                'unsup-loss {unsup_loss:.4f}',
                'mix-loss {mix_loss:.4f}',
                'train-iou {train_iou:.4f}',
            ]).format(
                it = it+1,
                max_it = self.args.max_iter,
                lr = lr,
                time = t_intv,
                eta = eta,
                total_loss=total_loss_avg,
                unsup_loss=loss_unsup_avg,
                mix_loss=loss_mix_avg,
                sup_loss=loss_seg_avg,
                train_iou=iou_train_avg,
            )
        if self.args.unsup_loss == 'adapt_focal':
            adapt_ratio_info = ', '+'adapt_ratio: %.2f'%(adapt_ratio_avg)
            solid_ratio_info = ', '+'solid_ratio: %.2f'%(solid_ratio_avg)
            msg = msg + adapt_ratio_info + solid_ratio_info
            for i in range(self.n_classes):
                tmp_info = ', ' + str(train_id_list[i]) + ': %.2f'%(self.criteria_unsup.cls_thresh[i])
                msg = msg + tmp_info
        logger.info(msg)
        self.iou_train_list = []
        self.loss_seg_list = []
        self.loss_mix_list = []
        if self.args.paste_mode == 'Dual_soft':
            self.loss_mix_src_list = []
            self.loss_mix_tar_list = []
        if self.args.unsup_loss == 'adapt_focal':
            self.adapt_ratio_list,self.adapt_ratio16_list,self.adapt_ratio32_list = [],[],[]
            self.solid_ratio_list,self.solid_ratio16_list,self.solid_ratio32_list = [],[],[]
        self.loss_unsup_list =[]
        self.total_loss_list = []
        self.start_time = ed
        return writer, logger

    #save the RGB image in tensor format 
    def save_tensor_image(self, tensor_img, file_path):
        #tensor_img: (1,3,H,W), torch.tensor
        assert len(tensor_img.shape)==4 and tensor_img.shape[0] == 1
        tensor_img = tensor_img.clone().detach()
        pil_img = self.to_pil(tensor_img.squeeze(0).cpu())
        pil_img.save(file_path)

    #save the grey label in tensor format
    def colorize_mask(self, mask):
        #mask tensor: (1,1,H,W), torch.tensor
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
                0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask