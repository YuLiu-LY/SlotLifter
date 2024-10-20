# SlotLifter
<p align="left">
    <a href='https://arxiv.org/abs/2408.06697'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://arxiv.org/pdf/2408.06697'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://slotlifter.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This repository contains the official implementation of the ECCV 2024 paper:

[SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](https://arxiv.org/abs/2408.06697)

[YuLiu](https://yuliu-ly.github.io)\*，[Baoxiong Jia](https://buzz-beater.github.io)\*，[Yixin Chen](https://yixchen.github.io), [Siyuan Huang](https://siyuanhuang.com)
<br>
<p align="center">
    <img src="assets/overview.png"> </img>
</p> 

## Environment Setup
We provide all environment configurations in ``requirements.txt``. To install all packages, you can create a conda environment and install the packages as follows: 
```bash
conda create -n slotlifter python=3.8
conda activate slotlifter
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
In our experiments, we used NVIDIA CUDA 12.1 on Ubuntu 22.04. Similar CUDA version should also be acceptable with corresponding version control for ``torch`` and ``torchvision``.

## Dataset
### 1. ShapeStacks, ObjectsRoom, CLEVRTex, Flowers
Download ShapeStacks, ObjectsRoom, CLEVRTex and Flowers datasets with
```bash
chmod +x scripts/downloads_data.sh
./downloads_data.sh
```
For ObjectsRoom dataset, you need to run ``objectsroom_process.py`` to save the tfrecords dataset as a png format.
Remember to change the ``DATA_ROOT`` in ``downloads_data.sh`` and ``objectsroom_process.py`` to your own paths.
### 2. PTR, Birds, Dogs, Cars
Download PTR dataset following instructions from http://ptr.csail.mit.edu. Download CUB-Birds, Stanford Dogs, and Cars datasets from [here](https://drive.google.com/drive/folders/1zEzsKV2hOlwaNRzrEXc9oGdpTBrrVIVk), provided by authors from [DRC](https://github.com/yuPeiyu98/DRC). We use the ```birds.zip```, ```cars.tar``` and ```dogs.zip``` and then uncompress them.

### 4. YCB, ScanNet, COCO
YCB, ScanNet and COCO datasets are available from [here](https://www.dropbox.com/sh/u1p1d6hysjxqauy/AACgEh0K5ANipuIeDnmaC5mQa?dl=0), provided by authors from [UnsupObjSeg](https://github.com/vLAR-group/UnsupObjSeg).

### 5. Data preparation
Please organize the data following [here](./data/README.md) before experiments.

## Training

To train the model from scratch we provide the following model files:
 - ``train_trans_dec.py``: transformer-based model
 - ``train_mixture_dec.py``: mixture-based model
 - ``train_base_sa.py``: original slot-attention
We provide training scripts under ``scripts/train``. Please use the following command and change ``.sh`` file to the model you want to experiment with. Take the transformer-based decoder experiment on Birds as an exmaple, you can run the following:
```bash
$ cd scripts
$ cd train
$ chmod +x trans_dec_birds.sh
$ ./trans_dec_birds.sh
```
Remember to change the paths in ``path.json`` to your own paths.
## Reloading checkpoints & Evaluation

To reload checkpoints and only run inference, we provide the following model files:
 - ``test_trans_dec.py``: transformer-based model
 - ``test_mixture_dec.py``: mixture-based model
 - ``test_base_sa.py``: original slot-attention

Similarly, we provide testing scripts under ```scripts/test```. We provide transformer-based model for real-world datasets (Birds, Dogs, Cars, Flowers, YCB, ScanNet, COCO) 
and mixture-based model for synthetic datasets(ShapeStacks, ObjectsRoom, ClevrTex, PTR). We provide all checkpoints [here](https://drive.google.com/drive/folders/10LmK9JPWsSOcezqd6eLjuzn38VdwkBUf?usp=sharing). Please use the following command and change ``.sh`` file to the model you want to experiment with:
```bash
$ cd scripts
$ cd test
$ chmod +x trans_dec_birds.sh
$ ./trans_dec_birds.sh
```

## Citation
If you find our paper and/or code helpful, please consider citing:
```
@inproceedings{Liu2024slotlifter,
  title={SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields},
  author={Liu, Yu and Jia, Baoxiong and Chen, Yixin and Huang, Siyuan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgement
This code heavily used resources from [PanopticLifting](https://github.com/nihalsid/panoptic-lifting), [BO-QSA](https://github.com/YuLiu-LY/BO-QSA), [SLATE](https://github.com/singhgautam/slate), [OSRT](https://github.com/stelzner/osrt), [IBRNet](https://github.com/googleinterns/IBRNet). We thank the authors for open-sourcing their awesome projects.
