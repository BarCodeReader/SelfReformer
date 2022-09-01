<img src="https://github.com/BarCodeReader/SelfReformer/blob/main/asset/logo.png" alt="drawing" width="1200"/>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/selfreformer-self-refined-network-with/salient-object-detection-on-dut-omron-2)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron-2?p=selfreformer-self-refined-network-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/selfreformer-self-refined-network-with/salient-object-detection-on-duts-te-1)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te-1?p=selfreformer-self-refined-network-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/selfreformer-self-refined-network-with/salient-object-detection-on-ecssd-1)](https://paperswithcode.com/sota/salient-object-detection-on-ecssd-1?p=selfreformer-self-refined-network-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/selfreformer-self-refined-network-with/salient-object-detection-on-hku-is-1)](https://paperswithcode.com/sota/salient-object-detection-on-hku-is-1?p=selfreformer-self-refined-network-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/selfreformer-self-refined-network-with/salient-object-detection-on-pascal-s-1)](https://paperswithcode.com/sota/salient-object-detection-on-pascal-s-1?p=selfreformer-self-refined-network-with)

# SelfReformer-pytorch

### Prerequisites
Ubuntu 18.04\
Python==3.8.3\
Torch==1.8.2+cu111\
Torchvision==0.9.2+cu111\
kornia

Besides, we use [git-lfs](https://git-lfs.github.com/) for large file management. Please also install it otherwise the .pth file might not be correctly downloaded.

### Dataset
For all datasets, they should be organized in below's fashion:
```
|__dataset_name
   |__Images: xxx.jpg ...
   |__Masks : xxx.png ...
```
For training, put your dataset folder under:
```
dataset/
```
For evaluation, download below datasets and place them under:
```
dataset/benchmark/
```
Suppose we use DUTS-TR for training, the overall folder structure should be:
```
|__dataset
   |__DUTS-TR
      |__Images: xxx.jpg ...
      |__Masks : xxx.png ...
   |__benchmark
      |__ECSSD
         |__Images: xxx.jpg ...
         |__Masks : xxx.png ...
      |__HKU-IS
         |__Images: xxx.jpg ...
         |__Masks : xxx.png ...
      ...
```
[**ECSSD**](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) || [**HKU-IS**](https://i.cs.hku.hk/~gbli/deep_saliency.html) || [**DUTS-TE**](http://saliencydetection.net/duts/) || [**DUT-OMRON**](http://saliencydetection.net/dut-omron/) || [**PASCAL-S**](http://cbi.gatech.edu/salobj/)

### Train & Test
**Firstly, make sure you have enough GPU RAM**.\
With default setting (batchsize=16), 24GB RAM is required, but you can always reduce the batchsize to fit your hardware.

Default values in option.py are already set to the same configuration as our paper, so \
after setting the ```--dataset_root``` flag in **option.py**, to train the model (default dataset: DUTS-TR), simply:
```
python main.py --GPU_ID 0
```
to test the model located in the **ckpt** folder (default dataset: DUTS-TE), simply:
```
python main.py --test_only --pretrain "bal_bla.pt" --GPU_ID 0
```
If you want to train/test with different settings, please refer to **option.py** for more control options.\
Currently only support training on single GPU.

### Pretrain Model & Pre-calculated Saliency Map
Pre-calculated saliency map: [[Google]](https://drive.google.com/file/d/1HPZvAFBS_5RaXggd4QqGFWQQs61cQ6m0/view?usp=sharing)

Pre-trained model on DUTS-TE: [[Google]](https://drive.google.com/file/d/19kO-IjZS56rIDTfscABhErTO-SP7oP4L/view?usp=sharing)

### Evaluation
Firstly, obtain predictions via
```
python main.py --test_only --pretrain "xxx/bal_bla.pt" --GPU_ID 0 --save_result --save_msg "abc"
```
Output will be saved in `./output/abc` if you specified the **save_msg** flag.

For *PR curve* and *F curve*, we use the code provided by this repo: [[BASNet, CVPR-2019]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)\
For *MAE*, *F measure*, *E score* and *S score*, we use the code provided by this repo: [[F3Net, AAAI-2020]](https://github.com/weijun88/F3Net#evaluation)

### Evaluation Results
#### Quantitative Evaluation
<img src="https://github.com/BarCodeReader/SelfReformer/blob/main/asset/table1.PNG" alt="drawing" width="1200"/>
<img src="https://github.com/BarCodeReader/SelfReformer/blob/main/asset/pr_curve.png" alt="drawing" width="1200"/>

#### Qualitative Evaluation
<img src="https://github.com/BarCodeReader/SelfReformer/blob/main/asset/figure1.png" alt="drawing" width="1200"/>

### License
Currently under **Attribution 4.0 International (CC BY 4.0)**

### Contribution
If you want to contribute or make the code better, simply submit a Pull-Request to **develop** branch.
