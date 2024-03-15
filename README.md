# PRISM
PRISM: A Promptable and Robust Interactive Segmentation Model with Visual Prompts

**News**

This repository is under construction

The [pretrained PRISM](https://drive.google.com/drive/u/1/folders/1B6Df44Gd9PEBGPkE1FwC8Ds4jefCekUB) models and [datasets](https://drive.google.com/drive/folders/13uGNb2WQhSQcBQIUhnvYJere1LBYGDsW?usp=sharing) are uploaded.

**TODO**

reduce the number of arguments

**Datasets**

We used four public [datasets](https://drive.google.com/drive/folders/13uGNb2WQhSQcBQIUhnvYJere1LBYGDsW?usp=sharing) for colon, pancreas, liver and kidney tumor segmentation. Here are the links for the original datasets




**Installation**
```
conda create -n prism python=3.9
conda activate prism
(Optional): sudo install git
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 # install pytorch
pip install git+https://github.com/facebookresearch/segment-anything.git # install segment anything packages
pip install git+https://github.com/deepmind/surface-distance.git # for normalized surface dice (NSD) evaluation
pip install -r requirements.txt
```


**Train**

```
python train.py --data colon --data_dir your_data_directory --max_epoch 200 --save_name test123 --num_clicks 50 --num_clicks_validation 10 --iter_nums 11 --multiple_outputs --dynamic --use_box --refine
```


**Train (Distributed Data Parallel)**

```
python train.py --data colon --data_dir your_data_directory --max_epoch 200 --save_name test123 --num_clicks 50 --num_clicks_validation 10 --iter_nums 11 --multiple_outputs --dynamic --use_box --refine --ddp
```
the only difference between this and above command is the use of "--ddp".



**Test**

put downloaded pretrained model under the implementation directory
```
python test.py --data colon --data_dir your_data_directory --split test --checkpoint best --save_name prism_pretrain --num_clicks 1 --iter_nums 11 --multiple_outputs --use_box --use_scribble --efficient_scribble --refine --refine_test
```





**FAQ**

if you got the error as AttributeError: module 'cv2' has no attribute 'ximgproc', please check [this](https://stackoverflow.com/questions/57427233/module-cv2-cv2-has-no-attribute-ximgproc) out
