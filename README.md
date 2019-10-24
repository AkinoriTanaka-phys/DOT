# DOT
code submission to NeurIPS2019 of the paper "Discriminator optimal transport".

# Environment
DL framework for python is `chainer`.
GPU is necessary.
In addition, `torch` and `torchvision` are required to execute `load_dataset.py` if you want to download CIFAR-10 or STL-10 dataset, and `tensorflow` is also necessary to execute `download.py` downloading inception model to measure inception score and FID.

# Demos
Demos can be excuted by
 1. `2d_demo.ipynb`
 2. `execute_dot.py`
 3. `execute_dot_cond.py`

## 1. `2d_demo.ipynb`:
This notebook executes 2d experiment on DOT.
All necessary files are included.

## 2. `execute_dot.py`:
The python script file execute latent space DOT for CIFAR-10 and STL-10 using trained models.
### preliminary
To run it, we need to prepare trained models by `train_GAN.py`.
#### data download for the training:
Please follow the next steps:
```
$ python load_dataset.py --root torchdata/ --data cifar10
$ python load_dataset.py --root torchdata/ --data stl10
$ python load_stl_to48.py --gpu 0
$ python download.py --outfile metric/inception_score.model
$ python get_mean_cov_2048featurespace.py --data CIFAR --gpu 0
$ python get_mean_cov_2048featurespace.py --data STL48 --gpu 0
$ mkdir scores
```
The first 3 executions download and prepare `training_data/CIFAR.npy`, `training_data/STL96.npy` and `training_data/STL48.npy`.
The following 3 executions download inception model and prepare data for calculating FID.

#### GAN training
After the above executions, we can execute `train_GAN.py` with OPTIONS:
> `--gpu` : GPU id<br>
> `--net` : Neural net architecture to train. `DCGAN` or `Resnet` <br>
> `--mode` : Neural net additional info. `SAGAN` or `SNGAN` or `WGAN-GP`<br>
> `--data` : Training data. `CIFAR` or `STL48`<br>
> `--objective` : Objective functions of the GAN. For `SAGAN` and `SNGAN`, we reccomend to use`NonSaturating` or `Hinge`. For `WGAN-GP`, please set `Wasserstein`<br>
> `--iters` : Total number of the iterations. `int`<br>
> `--report` : Number of the report step. `int`<br>
For example, 
```
$ python train_GAN.py --net DCGAN --data CIFAR --mode SAGAN --objective NonSaturating --iters 150000 --report 10000 --gpu 0
```
generates files named `DCGAN_G_CIFAR_SAGAN_NonSaturating_xx.npz` and `DCGAN_D_CIFAR_SAGAN_NonSaturating_xx.npz` to `trained_models/` if the inception score in each `10000` iteration exceeds the best score in the updating history. The number `xx` means the number of iteration of the saved models.

### DOT
`execute_dot.py` execute the latent space DOT with OPTIONS
> `--gpu` : gpu id<br>
> `--G` : Generator filename in trained_models/<br>
> `--D` : Discriminator filename in trained_models/<br>
> `--transport` : dot or naive<br>
> `--lr` : float for SGD updates<br>
> `--N_update` : the number of SGD update of each sample by DOT<br>
> `--showing_period` : the period for log-file under scores/<br>
> `--k` : if 1, k_eff in DOT is fixed 1. if not specified, k_eff is calculated.<br>
For example,
```
python execute_dot.py --G DCGAN_G_CIFAR_SAGAN_NonSaturating_140000.npz --D DCGAN_D_CIFAR_SAGAN_NonSaturating_140000.npz --transport dot --lr 0.01 --N_update 10 --showing_period 5 --gpu 0
```
executes latent space DOT by using specified models in `trained_models/` by applying `10` times SGD with `lr 0.01`.
The log of IS and FID will be written under scores/ every `showing_period` update by using 50000 samples.

## 3. `execute_dot_cond.py`:
The python script file execute conditional DOT.
To run it, we need
```
ResNetGenerator_850000.npz
SNResNetProjectionDiscriminator_850000.npz
```
in the directory trained_models/
The npz data is in public: https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi

In addition to it, please clone directories
>https://github.com/pfnet-research/sngan_projection/tree/master/gen_models<br>
>https://github.com/pfnet-research/sngan_projection/tree/master/dis_models<br>
>https://github.com/pfnet-research/sngan_projection/tree/master/source

to this directory.

The inception model which is explained in 2 also needed, and as a final requirement, we need mean and cov in inception model 2,048 feature space of the imagenet dataset. After calculating it, save them to
```
imagenet_inception_mean.npy
imagenet_inception_cov.npy
```
This step van be skipped just by using cifar's features by using option `--data cifar` of this script.
OPTIONS are same as 2.
