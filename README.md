# Threshold-adaptive unsupervised focal loss for domain adaptive semantic segmentation

Official implementation of "Threshold-adaptive unsupervised focal loss for domain adaptive semantic segmentation". Submitted to T-ITS on April 2, 2022.

[https://arxiv.org/abs/2208.10716](https://arxiv.org/abs/2208.10716)

## Abstract

Semantic segmentation is an important task for intelligent vehicles to understand the environment. Current deep learning based methods require large amounts of labeled data for training. Manual annotation is expensive, while simulators can provide accurate annotations. However, the performance of the semantic segmentation model trained with synthetic datasets will significantly degenerate in the actual scenes. Unsupervised domain adaptation (UDA) for semantic segmentation is used to reduce the domain gap and improve the performance on the target domain. Existing adversarial-based and self-training methods usually involve complex training procedures, while entropy-based methods have recently received attention for their simplicity and effectiveness. However, entropy-based UDA methods have problems that they barely optimize hard samples and lack an explicit semantic connection between the source and target domains. In this paper, we propose a novel two-stage entropy-based UDA method for semantic segmentation. In stage one, we design a threshold-adaptative unsupervised focal loss to regularize the prediction in the target domain. It firstly introduces unsupervised focal loss into UDA for semantic segmentation, helping to optimize hard samples and avoiding generating unreliable pseudo labels in the target domain. In stage two, we employ cross-domain image mixing (CIM) to bridge the semantic knowledge between two domains and incorporate longtail class pasting to alleviate the class imbalance problem. Extensive experiments on synthetic-to-real and cross-city benchmarks demonstrate the effectiveness of our method. It achieves state-of-the-art performance using DeepLabV2, as well as competitive performance using the lightweight BiSeNet with great advantages in training and inference time.

## Environment Setup

I verified this reporitory in Ubuntu18.04 with Anaconda3, Pytorch 1.8.0, two 1080Ti GPUs.

First, create the environment named uda and activate the uda environment through:

```
conda create -n uda python=3.8 -y
conda activate uda
```

Then install the required packages though:

```
pip install -r requirements.txt 
```

Download the code from github and change the directory:

```
git clone https://github.com/ywher/TUFL
cd TUFL
```

## Dataset preparation

We only show how to set the Cityscapes-to-CrossCity setting. You can prepare the SYNTHIA-to-Cityscapes and GTA5-to-Cityscapes similarly.

Download Cityscapes and CrossCity dataset, then organize the folder as follows:

```
|TUFL/data
│     ├── cityscapes/   
|     |   ├── gtFine/
|     |   |   ├── train/
|     |   |   ├── val/
|     |   |   ├── test/
|     |   ├── leftImg8bit/
│     ├── NTHU/
|     |   ├── Rio/
|     |   |   ├── Images/
|     |   |   ├── Labels/
|     |   ├── Rome/
|     |   ├── Taipei/
|     |   ├── Tokyo/
      ...
```

## Training and Evaluation example

We use two 1080Ti GPUs for training and one of them for evaluation.

### Train with unsupervised focal loss in Cityscapes-to-Rio setting

First, you need to download the pretrained model of BiSeNet on Cityscapes and put it in the pretrained folder.  (You can also re-train the BiSeNet on Cityscapes.)

(Link: [https://drive.google.com/file/d/1P0G1mcNomCxUqMScKJlzSDpvB9vDf-6u/view?usp=sharing](https://drive.google.com/file/d/1P0G1mcNomCxUqMScKJlzSDpvB9vDf-6u/view?usp=sharing))

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
--supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes' \
--freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
--warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Rio' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08
```

This training setting would take about 14 minutes using two 1080Ti GPUs. The results will be saved in the TUFL/outputs.

You can refer to TUFL/unsup/parse_args.py for more information about the training setting. More training instructions can be found in train_crosscity.sh

### Evaluation in Cityscapes-to-Rio setting

First, you can download our trained models from Cityscapes to Rio and put them in the **TUFL/outputs**.

(Link: [https://drive.google.com/drive/folders/1YAcjbf7Lt4iER3K-idP_olONO1ybRCP5?usp=sharing](https://drive.google.com/drive/folders/1YAcjbf7Lt4iER3K-idP_olONO1ybRCP5?usp=sharing))

You need to specify the --save_path (model path) and --city (target city name) in the evaluation. Note that the --save_path should be in the **TUFL/outputs** folder.

```
CUDA_VISIBLE_DEVICES=0 python3 ./utils/evaluate.py --dataset 'CrossCity' --dataset_mode 'test' --iou_mode 'cross' --simple_mode \
--save_pth 'CityScapes_BiSeNet_2k_Rio_unsup_focal_0.8_0.08_ohem_0.5/model_epoch_2000.pth' --city 'Rio'
```

More trained models can be found in the following links:

* Cityscapes-to-Rome: [https://drive.google.com/drive/folders/1CO_SxoiLP1lIqm4fBYk1SQnPG1wZNOBn?usp=sharing](https://drive.google.com/drive/folders/1CO_SxoiLP1lIqm4fBYk1SQnPG1wZNOBn?usp=sharing)
* Cityscapes-to-Taipei: [https://drive.google.com/drive/folders/1byndx9ykhg1hOQNlVLqn9znidnj0Rf5C?usp=sharing](https://drive.google.com/drive/folders/1byndx9ykhg1hOQNlVLqn9znidnj0Rf5C?usp=sharing)
* Cityscapes-to-Tokyo: [https://drive.google.com/drive/folders/1NXTiTlXA3pvLVXYcPUgtLZULTqiIqI1W?usp=sharing](https://drive.google.com/drive/folders/1NXTiTlXA3pvLVXYcPUgtLZULTqiIqI1W?usp=sharing)

The results of our trained models are listed in the following. The ss and ms mean single scale and multi scale testing respectively.

| Setting\iteration (mIoU) | 2k iteration(ss/ms) | best iteration     |
| ------------------------ | ------------------- | ------------------ |
| Cityscapes-to-Rome       | 52.27/53.35         | 53.30/53.96 (1.2k) |
| Cityscapes-to-Rio        | 56.39/58.02         | 56.49/58.53 (1.8k) |
| Cityscapes-to-Taipei     | 51.05/52.15         | 51.58/52.65 (1.4k) |
| Cityscapes-to-Tokyo      | 48.59/59.70         | 50.01/51.27 (1k)   |

# License

Some of the code is borrowed from [BiSeNet](https://github.com/CoinCheung/BiSeNet)

If you use this code in your research please consider citing our work

```

```
