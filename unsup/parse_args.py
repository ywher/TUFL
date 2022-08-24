import argparse

###define the config parameters for training
def parse_args():
    parse = argparse.ArgumentParser()
    #training parameters
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=0)
    parse.add_argument('--n_workers', dest='n_workers', type=int, default=8)
    parse.add_argument('--msg_iter', dest='msg_iter', type=int, default=20, help='log per iteration')
    parse.add_argument('--max_iter', dest='max_iter', type=int, default=20000, help='total training iterations')
    parse.add_argument('--warm_up_ratio', dest='warm_up_ratio', type=float, default=0.05, help='warm up iterations of total')
    parse.add_argument('--n_img_per_gpu_train', dest='n_img_per_gpu_train', type=int, default=1, help='number of train images per gpu')
    parse.add_argument('--n_img_per_gpu_unsup', dest='n_img_per_gpu_unsup', type=int, default=1, help='number of target images per gpu')
    parse.add_argument('--freeze_bn', dest='freeze_bn', action='store_true', help='freeze the bn layder, model.eval()')
    parse.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parse.add_argument('--supervision_mode', dest='supervision_mode',type=str, default='unsup', choices=['sup', 'unsup_single', 'unsup'], help='training mode')
    #learning rate
    parse.add_argument('--lr_start', help='start learning rate', dest='lr_start', type=float,default=0.5e-5)
    parse.add_argument('--warmup_start_lr', help='warmup start learning rate', dest='warmup_start_lr', type=float, default=1e-5)
    parse.add_argument('--lr_power', help='ploy learning rate power', dest='lr_power', type=float, default=0.9)
    parse.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parse.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
    #source dataset
    parse.add_argument('--dataset', dest='dataset', type=str, default='CityScapes',
        choices=['CityScapes_Rome', 'CityScapes_Rio', 'CityScapes_Tokyo', 'CityScapes_Taipei', 'Mapillary', 'CityScapes'], help='source dataset')
    parse.add_argument('--n_classes', dest='n_classes', type=int, default=13, choices=[13,16,19])
    parse.add_argument('--target_dataset', dest='target_dataset', type=str, default='CrossCity', choices=['CityScapes', 'CrossCity'], help='target dataset')
    parse.add_argument('--target_city', dest='target_city', type=str, default='Rome', choices=['all', 'Rome', 'Rio', 'Tokyo', 'Taipei'], help='city in CrossCity')
    parse.add_argument('--ignore_index', dest='ignore_index', type=int, default=255, help='ignore index in the dataset')
    parse.add_argument('--sup_ratio', type=float, default=1.0, help='the ratio of dataset being used in supervised training')
    #pretrained dataset and model
    parse.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default=None, 
        choices=['CityScapes', 'Mapillary', None, 'CityScapes_Rome', 'CityScapes_Rio', 'CityScapes_Tokyo', 'CityScapes_Taipei'])
    parse.add_argument('--pretrain_drop_last_layer', dest='pretrain_drop_last_layer', action='store_true', help='drop the classification layer of the pretrained model')
    parse.add_argument('--sup_mode', dest='sup_mode', type=str, default='deep', choices=['deep', 'last'], help='model sup mode, deep or last')
    parse.add_argument('--segmentation_model', dest='segmentation_model', type=str, default='BiSeNet', choices=['BiSeNet'])
    parse.add_argument('--pretrained_path', dest='pretrained_path', type=str, default='None', help='the path to load pretrained model for net')
    parse.add_argument('--pesudo_pretrained_path', dest='pesudo_pretrained_path', type=str, default='None', help='the path to load pretrained model for pseudo net')
    #loss function
    parse.add_argument('--weight_mode', dest='weight_mode', type=str, default='middle', choices=['small','middle','large'], help='the class weight for loss function') 
    parse.add_argument('--sup_loss', dest='sup_loss', type=str, default='regular', choices=['regular', 'ohem'], help='source domain supervised loss function')
    parse.add_argument('--unsup_single_loss', dest='unsup_single_loss', type=str, default=None,
        choices=[None, 'entropy', 'square', 'confi_entropy'], help='target domain unsupervised loss (single target image)')
    parse.add_argument('--unsup_loss', dest='unsup_loss', type=str, default='crossentropy', 
        choices=[None,'crossentropy','focal','adapt_focal'], help='target domain unsupervised loss (disturbed imags pairs)')
    parse.add_argument('--uda_confidence_thresh', dest='uda_confidence_thresh',type=float, default=0.8, help='confidence threshold for unsupervised loss')
    parse.add_argument('--uda_softmax_temp', dest='uda_softmax_temp',type=float, default=-1)
    parse.add_argument('--unsup_coeff', dest='unsup_coeff',type=float, default=1e-2, help='unsupervised loss coefficient')
    parse.add_argument('--focal_gamma', dest='focal_gamma', type=float, default=2, help='the gamma for unsupervised focal loss')
    parse.add_argument('--adapt_a', dest='adapt_a', type=float, default=0.9, help='adapt focal a')
    parse.add_argument('--a_decay', dest='a_decay', type=float, default=0, help='adapt focal a_decay')
    parse.add_argument('--adapt_b', dest='adapt_b', type=float, default=0.8, help='adapt focal b')
    parse.add_argument('--adapt_d', dest='adapt_d', type=float, default=8.0, help='adapt focal d')
    parse.add_argument('--loss_ohem', dest='loss_ohem', action='store_true', help='whether to use the ohem in unsupervised loss')
    parse.add_argument('--loss_ohem_ratio', dest='loss_ohem_ratio', type=float, default=0.25, help='the samples with top r% losses are selected')
    
    #cross domain image mixing
    parse.add_argument('--paste_mode', dest='paste_mode', type=str, default='None', 
        choices=['None','Single','Dual_soft'], help='how to paste the images from two domains')
    parse.add_argument('--mixed_weight', dest='mixed_weight', type=float, default=1, help='mixed loss coefficient')
    parse.add_argument('--mixed_boundary', dest='mixed_boundary', type=int, default=0, help='the highlight boundary width')
    parse.add_argument('--mixed_gaussian_kernel', dest='mixed_gaussian_kernel', type=int, default=0, help='the gaussian blur kernel on the boundary')
    parse.add_argument('--long_tail', dest='long_tail', action='store_true', help='use the long tail past')
    parse.add_argument('--long_tail_p', dest='long_tail_p', type=float, default=0.3, help='the probability to use long tail paste')
    parse.add_argument('--aug_mix', dest='aug_mix', type=bool, default=False, help='augment the mixed image and label')
    parse.add_argument('--class_relation', dest='class_relation', type=bool, default=False, help='use the class relation to pick the classes')
    parse.add_argument('--bapa_boundary', dest='bapa_boundary', type=int, default=0, help='the ditance param lambda_d of bapa')
    parse.add_argument('--soft_weight', dest='soft_weight', type=float, default=0.8, help='soft paste mode ratio')
    parse.add_argument('--src_mix_weight', dest='src_mix_weight', type=float, default=1, help='the source mixed loss weight in dual paste')
    parse.add_argument('--tar_mix_weight', dest='tar_mix_weight', type=float, default=1, help='the target mixed loss weight in dual paste')

    return parse.parse_args()