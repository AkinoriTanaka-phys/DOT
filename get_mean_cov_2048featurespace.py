import argparse
import os

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
import numpy as np
from evaluation import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', type=str, default='CIFAR')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = Inception()
    serializers.load_hdf5('metric/inception_score.model', model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    datapath = 'training_data/{}.npy'.format(args.data)
    mean_savepath = 'metric/{}_inception_mean.npy'.format(args.data)
    cov_savepath = 'metric/{}_inception_cov.npy'.format(args.data)

    img = 255*xp.load(datapath).astype(xp.float32)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(model, img)

    np.save(mean_savepath, mean)
    np.save(cov_savepath, cov)

