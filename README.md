# ECO-pytorch

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

> Note that:
> 1. The dataset should be organized as: <dataset_frames_root_path>/<video_name>/<frame_images>
> 2. If your <frame_images> filename contains prefix, you should specific that using "--rgb_prefix" argument

## Training

[UCF101 - ECO - RGB] command:

```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
        --arch ECO --num_segments 4 --gd 5 --lr 0.001 --lr_steps 30 60 --epochs 80 \
        -b 32 -i 4 -j 2 --dropout 0.5 --snapshot_pref ucf101_ECO --rgb_prefix img_ \
        --consensus_type identity --eval-freq 1
```