#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 ./utils/evaluate.py --dataset 'CrossCity' --dataset_mode 'test' --iou_mode 'cross' --simple_mode \
--save_pth 'CityScapes_BiSeNet_2k_Rio_unsup_focal_0.8_0.08_ohem_0.5/model_epoch_2000.pth' --city 'Rio'


