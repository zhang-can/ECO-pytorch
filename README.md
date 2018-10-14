#### This is a fork of [Can Zhang](https://github.com/zhang-can/ECO-pytorch)'s PyTorch implementation for the [paper](https://arxiv.org/pdf/1804.09066.pdf):
##### " ECO: Efficient Convolutional Network for Online Video Understanding, European Conference on Computer Vision (ECCV), 2018." By Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox
 
 
##### NOTE
* Trained models on Kinetics dataset for ECO Lite and C3D are provided. 
* **Keep watching for updates**

* Pre-trained model for 2D-Net is provided by [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch).


### Environment:
* Python 3.5.2
* PyTorch 0.4.1
* TorchVison: 0.2.1

### Clone this repo

```
git clone https://github.com/mzolfaghari/ECO-pytorch
```

### Generate dataset lists

```bash
python gen_dataset_lists.py <ucf101/something> <dataset_frames_root_path>
```
e.g. python gen_dataset_lists.py something ~/dataset/20bn-something-something-v1/

> The dataset should be organized as:<br>
> <dataset_frames_root_path>/<video_name>/<frame_images>

### Training
1. Download the initialization and trained models:

```Shell
      sh models/download_models.sh
```

* If you can not access Google Drive, please download the pretrained models from [BaiduYun](https://pan.baidu.com/s/1Hx52akJLR_ISfX406bkIog), and put them in "models" folder.

2. Command for training ECO Lite model:

```bash
    ./scripts/run_ECOLite_kinetics.sh local
```

3. For training C3D network use the following command:

```bash
    ./scripts/run_c3dres_kinetics.sh local
```

4. For finetuning on UCF101 use the following command:

```bash
    ./scripts/run_ECOLite_finetune_UCF101.sh local
```

### NOTE
* If you want to train your model from scratch change the config as following:
```bash
    --pretrained_parts scratch
```
* configurations explained in "opts.py"

#### TODO
1. ECO Full
2. Trained models on other datasets


#### Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@inproceedings{ECO_eccv18,
author={Mohammadreza Zolfaghari and
               Kamaljeet Singh and
               Thomas Brox},
title={{ECO:} Efficient Convolutional Network for Online Video Understanding},	       
booktitle={ECCV},
year={2018}
}
```

#### Contact

  [Mohammadreza Zolfaghari](https://github.com/mzolfaghari/ECO-pytorch), [Can Zhang](https://github.com/zhang-can/ECO-pytorch)

  Questions can also be left as issues in the repository. We will be happy to answer them.
