#shannon
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup_single' --unsup_single_loss 'entropy' --dataset 'CityScapes' --pretrained_dataset 'CityScapes_Rome' \
# --freeze_bn --max_iter 20000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --target_dataset 'CrossCity' --target_city 'Rome' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.25 --unsup_coeff 0.01 #0.04

#cross
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup' --unsup_loss 'crossentropy' --dataset 'CityScapes' --pretrained_dataset 'CityScapes_Rio' \
# --freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --target_dataset 'CrossCity' --target_city 'Rio' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.25 --unsup_coeff 0.01

#focal
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes' \
# --freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Rome' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes' \
# --freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Rio' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes' \
# --freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Taipei' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
--supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes' \
--freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
--warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Tokyo' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08

#adapt_focal
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup' --unsup_loss 'adapt_focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes_Rio' \
# --focal_gamma 2 --adapt_b 0.8 --adapt_a 0.9 --adapt_d 8.0 \
# --freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --target_dataset 'CrossCity' --target_city 'Rio' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08

#mix
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
# --supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes_Rio' \
# --paste_mode 'Single' --mixed_weight 0.5 \
# --freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
# --warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Rio' --n_classes 13 --unsup_coeff 0.08 \
# --pretrained_path '/media/fzx/1/yanweihao/BiSeNet-uda/outputs/crosscity/Rio/focal/CityScapes_BiSeNet_unsup_focal_0.8_0.08_ohem_0.5/model_epoch_1200.pth' \
# --pesudo_pretrained_path '/media/fzx/1/yanweihao/BiSeNet-uda/outputs/crosscity/Rio/focal/CityScapes_BiSeNet_unsup_focal_0.8_0.08_ohem_0.5/model_epoch_1200.pth'