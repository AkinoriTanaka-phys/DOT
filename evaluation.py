import numpy as np
import scipy


import chainer
from chainer import cuda
from chainer import datasets
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from inception_score import Inception
from inception_score import inception_score

import math

import cupy as xp
import DOT



def load_inception_model():
    model = Inception()
    serializers.load_hdf5('metric/inception_score.model', model)
    model.to_gpu()
    return model


## modified version of https://github.com/pfnet-research/chainer-gan-lib/blob/master/common/evaluation.py
# Copyright (c) 2017 pfnet-research
# Released under the MIT license
# https://github.com/pfnet-research/chainer-gan-lib/blob/master/LICENSE
def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp
    ys = xp.empty((n, 2048), dtype=xp.float32)

    for i in range(n_batches):
        print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = chainer.cuda.to_cpu(xp.mean(ys, axis=0))
    # cov = F.cross_covariance(ys, ys, reduce="no").data.get()
    cov = np.cov(chainer.cuda.to_cpu(ys).T)

    return mean, cov

def FID(m0,c0,m1,c1):
    ret = 0
    ret += np.sum((m0-m1)**2)
    ret += np.trace(c0 + c1 - 2.0*scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)

def calc_FID(img, model, data='CIFAR'):#, stat_file="%s/cifar-10-fid.npz"%os.path.dirname(__file__)):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    data_m = np.load("metric/{}_inception_mean.npy".format(data))
    data_c = np.load("metric/{}_inception_cov.npy".format(data))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(model, img)
    fid = FID(data_m, data_c, mean, cov)
    return fid

def calc_inception(gen, data='CIFAR'):
    @chainer.training.make_extension()
    def evaluation(trainer):
        model = load_inception_model()

        ims = []
        xp = gen.xp

        batchsize = 50
        n_img = 50000

        for i in range(0, n_img, batchsize):
            im = DOT.make_image(gen, None, batchsize, N_update=0, ot=False)
            im = np.asarray(np.clip(im * 127.5 + 127.5, 0.0, 255.0), dtype=np.float32)
            if i==0:
                ims = im
            else:
                ims = np.concatenate((ims, im))

        #if args.samples > 0:
        #    ims = ims[:args.samples]

        fid = calc_FID(ims, model, data)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            mean, std = inception_score(model, ims)

        chainer.reporter.report({
            'inception_mean': mean,
            'inception_std': std
        })

        chainer.reporter.report({
            'FID': fid
        })

    return evaluation

def save_models(G, D, net='Resnet', data='CIFAR', mode='SAGAN', objective='NonSaturating'):
    @chainer.training.make_extension()
    def save(trainer):
        G.to_cpu()
        D.to_cpu()
        serializers.save_npz("trained_models/{}_G_{}_{}_{}_{}.npz".format(net, data, mode, objective, trainer.updater.iteration), G)
        serializers.save_npz("trained_models/{}_D_{}_{}_{}_{}.npz".format(net, data, mode, objective, trainer.updater.iteration), D)
        G.to_gpu()
        D.to_gpu()
    return save
