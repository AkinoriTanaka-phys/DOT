# DOT
---
code submission to NeurIPS2019 of the paper "Discriminator optimal transport".

# Environment
---
DL framework for python is `chainer`.
GPU is necessary.

# Demos
---
Demos can be excuted by
 1. 2d_demo.ipynb
 2. execute_dot.py
 3. execute_dot_cond.py

## 1. 2d_demo.ipynb:
This notebook executes 2d experiment on DOT.
All necessary files are included within this supplementary materials.

## 2. execute_dot.py:
The python script file execute DOT using trained models of SAGAN which is included in this supplementary materials.
To run it, please download inception model following by
https://github.com/mattya/chainer-inception-score/tree/0c43b55b9bcba8149a9ed0b5d0bc4c5eceb49540
to metric/ directory by the filename "inception_score.model".
OPTIONS are
```
 --gpu : gpu id
 --model : only SAGAN available
 --transport : dot or naive
 --lr : float for SGD updates
 --N_update : the number of SGD update of each sample by DOT
 --showing_period : the period for log-file under scores/
```

For example,
```
$ python excute_dot.py --gpu 0 --model SAGAN --transport dot --lr 0.01 --N_update 30 --showing_period 30
```

executes latent space DOT by using SAGAN models by applying 30 times SGD with lr 0.01.
The log of IS and FID will be written under scores/ every "showing_period" update by using 50000 samples.

## 3. execute_dot_cond.py:
The python script file execute conditional DOT.
To run it, we need
```
ResNetGenerator_850000.npz
SNResNetProjectionDiscriminator_850000.npz
```
 
in the directory trained_models/
The npz data is in public: https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi

In addition to it, please clone directories
>https://github.com/pfnet-research/sngan_projection/tree/master/gen_models

>https://github.com/pfnet-research/sngan_projection/tree/master/dis_models

>https://github.com/pfnet-research/sngan_projection/tree/master/source

to this directory.

The inception model which is explained in 2 also needed, and as a final requirement, we need mean and cov in inception model 2,048 feature space of the imagenet dataset. After calculating it, save them to
```
imagenet_inception_mean.npy
imagenet_inception_cov.npy
```
This step van be skipped just by using cifar's features by using option --data cifar of this script.
OPTIONS are same as 2.