# ECO-pytorch

* This unofficial repo implements the ECO network structure with PyTorch, official repo is [here](https://github.com/mzolfaghari/ECO-efficient-video-understanding).
* Pre-trained model for 2D-Net is provided by [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch), and 3D-Net use the Kinetics-pretrained model of 3D-Resnet18 provided by [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).
* Codes modified from [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch).

> paper info:
> **"ECO: Efficient Convolutional Network for Online Video Understanding"**
> By Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox
> [paper link](https://arxiv.org/pdf/1804.09066.pdf)

## Environment:
* Python 3.6.4
* PyTorch 0.3.1

## Clone this repo

```
git clone https://github.com/zhang-can/ECO-pytorch
```

## Generate dataset lists

```bash
python gen_dataset_lists.py <ucf101/something> <dataset_frames_root_path>
```
e.g. python gen_dataset_lists.py something ~/dataset/20bn-something-something-v1/

```
The dataset should be organized as:
<dataset_frames_root_path>/<video_name>/<frame_images>
```

## Training

[UCF101 - ECO - RGB] command:

```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
        --arch ECO --num_segments 4 --gd 5 --lr 0.001 --lr_steps 30 60 --epochs 80 \
        -b 32 -i 4 -j 2 --dropout 0.5 --snapshot_pref ucf101_ECO --rgb_prefix img_ \
        --consensus_type identity --eval-freq 1
```