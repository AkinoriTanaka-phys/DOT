# DOT
This repository is for code submission to NeurIPS2019 of the paper "Discriminator optimal transport"([arXiv](http://arxiv.org/abs/1910.06832)).

# Environment
DL framework for python is `chainer`.

# Demos
Experiments in the paper can be excuted by
 1. `2d_demo.ipynb`
 2. `execute_dot.py`
 3. `execute_dot_cond.py`
 4. `execute_mh.py`

## 1. `2d_demo.ipynb`:
This notebook executes 2d experiment on DOT.
All necessary files are included.

## 2. `execute_dot.py`:
The python script file executes latent space DOT for CIFAR-10 and STL-10 using trained models.
Because of Licence issue, we cannot supply GAN training code here, sorry.
For example, please train your model by using the repository: https://github.com/pfnet-research/chainer-gan-lib

`execute_dot.py` executes the latent space DOT with OPTIONS
> `--gpu` : GPU id.<br>
> `--G` : Generator filename in `trained_models/`.<br>
> `--D` : Discriminator filename in `trained_models/`.<br>
> `--transport` : `dot` or `naive`.<br>
> `--optmode` : `adam` or `sgd`.<br>
> `--lr` : float for SGD updates.<br>
> `--N_update` : the number of SGD update of each sample by DOT.<br>
> `--showing_period` : the period for log-file under `scores/`.<br>
> `--k` : if 1, k_eff in DOT is fixed 1. if not specified, k_eff is calculated.<br>

For example,
```
$ python execute_dot.py --G DCGAN_G_CIFAR_SAGAN_NonSaturating_140000.npz --D DCGAN_D_CIFAR_SAGAN_NonSaturating_140000.npz --transport dot --optmode sgd --lr 0.01 --N_update 10 --showing_period 5 --gpu 0
```
executes latent space DOT by using the specified models in `trained_models/` by applying `0`, `5`, `10` times `sgd` with `lr 0.01`.
The log of IS and FID will be written under `scores/uncond_year_month_day_time.txt` by using 50000 samples.

## 3. `execute_dot_cond.py`:
The python script file executes conditional DOT.
To run it, we need
```
ResNetGenerator_850000.npz
SNResNetProjectionDiscriminator_850000.npz
```
in the directory `trained_models/`
The npz data is in public: https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi

In addition to it, please clone directories
>https://github.com/pfnet-research/sngan_projection/tree/master/gen_models<br>
>https://github.com/pfnet-research/sngan_projection/tree/master/dis_models<br>
>https://github.com/pfnet-research/sngan_projection/tree/master/source

to this directory.

The inception model which is explained in 2 also needed, and as a final requirement, we need mean and cov in inception model 2,048 feature space of the imagenet dataset. After calculating it, save them to
```
metric/imagenet_inception_mean.npy
metric/imagenet_inception_cov.npy
```
This step can be skipped just by using cifar's features by using option `--data CIFAR` of this script.
OPTIONS are same as 2-2, and the log-file will be saved to `scores/cond_year_month_day_time.txt`.

## 4. `execute_mh.py`:
`execute_mh.py` executes the re-implemented Metropolis-Hastings GAN sampling with OPTIONS
> `--gpu` : GPU id.<br>
> `--G` : Generator filename in `trained_models/`.<br>
> `--D` : Discriminator filename in `trained_models/`.<br>
> `--N_update` : the number of SGD update of each sample by DOT.<br>
> `--showing_period` : the period for log-file under `scores/`.<br>
> `--calib` : if True, calibration is applied to the discriminator.<br>
> `--initdata` : if True, data's certain minibatch is chosen as initial states of the Markov Chain. The update is repeated until generated data is accepted.<br>

For example,
```
python execute_mh.py --gpu 0 --G DCGAN_G_STL48_SAGAN_NonSaturating_140000.npz --D DCGAN_D_STL48_SAGAN_NonSaturating_140000.npz --calib True --initdata True --N_update 10 --showing_period 5
```
executes MH-GAN sampling by using the specified models in `trained_models/` with Markov chain length = `0`, `5`, `10` initialized by data itself, with calibrated discriminator.
The log of IS and FID will be written under `scores/MH_year_month_day_time.txt`.
