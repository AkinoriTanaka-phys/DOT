import argparse
import numpy as np
import scipy
from datetime import datetime

import chainer
from chainer import cuda
from chainer import datasets
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from metric.inception_score import Inception
from metric.inception_score import inception_score

from evaluation import *

import math

import cupy as xp
from model import *

# need to be downloaded models from https://github.com/pfnet-research/sngan_projection
from gen_models.resnet import ResNetGenerator
from dis_models.snresnet import SNResNetProjectionDiscriminator

from source.miscs.random_samples import sample_categorical, sample_continuous

import DOT_cond 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optmode', type=str, default='sgd')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--data', type=str, default='imagenet')
    parser.add_argument('--N_update', type=int, default=30)
    parser.add_argument('--lr', type=float, default=10**(-2)) 
    parser.add_argument('--showing_period', type=int, default=10)
    parser.add_argument('--transport', type=str, default='dot')
    parser.add_argument('--G', type=str, default="ResNetGenerator_850000.npz")
    parser.add_argument('--D', type=str, default="SNResNetProjectionDiscriminator_850000.npz")
    parser.add_argument('--k', type=int, default=None)
    return parser.parse_args()

###
def calc_scores(G, D, data, evmodel, dot_mode, N_update, batchsize=50, n_img=50000, k=1.0, lr=0.1):
    """ dot_mode = [target, latent, bare] """
    for i in range(0, n_img, batchsize):
        im = DOT_cond.make_image(G, D, batchsize, N_update=N_update, mode=dot_mode, k=k, lr=lr, optmode=args.optmode)
        im = np.asarray(np.clip(im * 127.5 + 127.5, 0.0, 255.0), dtype=np.float32)
        if i==0:
            ims = im
        else:
            ims = np.concatenate((ims, im))

    if args.samples > 0:
        ims = ims[:args.samples]

    fid = calc_FID(ims, evmodel, data=data)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(evmodel, ims)
    return fid, mean, std


def main(args, G, D, data, evmodel, k, transport, N_update, showing_period):
    lr = float(args.lr)
    if args.k==None:
         nk = xp.mean(k(xp.arange(1000)).data)
    else:
         nk = k
    filename="cond_" + datetime.now().strftime("%Y_%m_%d_%H%M%S")+".txt"
    with open("scores/{}".format(filename), "w") as fileobj:
        fileobj.write("{}\n".format(args.G))
        fileobj.write("{}\n".format(args.D))
        fileobj.write("DOTmode:{}\n".format(transport))
        fileobj.write("lr:{}\n".format(args.lr))
        fileobj.write("mean k:{}\n\n".format(cuda.to_cpu(nk)))
        for n_update in range(0, N_update+1, showing_period):
            fid, inception_mean, inception_std = calc_scores(G, D, data, evmodel, transport, n_update, k=k, lr=args.lr)
            fileobj.write("n_update:{}\n".format(n_update))
            fileobj.write("IS:{}pm{}\n".format(inception_mean, inception_std))
            fileobj.write("FID:{}\n\n".format(fid))

if __name__ == '__main__':
    args = parse_args()
    evmodel = Inception()
    serializers.load_hdf5('metric/inception_score.model', evmodel)

    G = ResNetGenerator(n_classes=1000)
    D = SNResNetProjectionDiscriminator(n_classes=1000)
    # available on https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi 
    serializers.load_npz("trained_models/" + args.G, G)
    serializers.load_npz("trained_models/" + args.D, D)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        evmodel.to_gpu()
        G.to_gpu()
        D.to_gpu()
    G, D = DOT_cond.thermalize_spectral_norm(G, D)

    if args.k==None:
        k = L.EmbedID(1000, 1, initialW=DOT_cond.return_ks(G, D, nlabels=1000))
        k.to_gpu()
    else:
        k = args.k*xp.ones([1])
    main(args, G, D, args.data, evmodel, k, transport=args.transport, N_update=args.N_update, showing_period=args.showing_period)

