import argparse
import numpy as np
import scipy
import os
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
import MH

from train_GAN import returnG, returnD
from execute_dot import load_GD

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--evmodel', type=str, default='metric/inception_score.model')
    parser.add_argument('--G', type=str, default='trained_models/xxxx')
    parser.add_argument('--D', type=str, default='trained_models/xxxx')
    parser.add_argument('--N_update', type=int, default=100)
    parser.add_argument('--showing_period', type=int, default=10)
    parser.add_argument('--calib', type=str, default='True')
    parser.add_argument('--initdata', type=str, default='True')
    return parser.parse_args()

def calc_scores(G, D, data, evmodel, C, N_update, batchsize=50, n_img=50000, init_data=False):
    for i in range(0, n_img, batchsize):
        if init_data == True:
            x_ini = (xp.load("training_data/{}.npy".format(data))).astype(xp.float32)*2 - 1
            xp.random.shuffle(x_ini)
            x_ini = x_ini[:batchsize]
        else:
            x_ini = None
        im, acceptance_rate = MH.make_image2(G, D, batchsize, C, N_update=N_update, initial=x_ini)
        im = np.asarray(np.clip(im * 127.5 + 127.5, 0.0, 255.0), dtype=np.float32)
        if i==0:
            ims = im
            accs = [acceptance_rate]
        else:
            ims = np.concatenate((ims, im))
            accs.append(acceptance_rate)
    acceptance = np.mean(accs)
    if args.samples > 0:
        ims = ims[:args.samples]
    fid = calc_FID(ims, evmodel, data=data)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(evmodel, ims)
    return fid, mean, std, acceptance

def main(args, G, D, data, evmodel, C, N_update, showing_period, init_data):
    filename="MH_" + datetime.now().strftime("%Y_%m_%d_%H%M%S")+".txt"
    with open("scores/{}".format(filename), "w") as fileobj:
        fileobj.write("{}\n".format(args.G))
        fileobj.write("{}\n".format(args.D))
        fileobj.write("Calibration:{}\n".format(C!=None))
        fileobj.write("initialize by real data:{}\n\n".format(init_data))
        for n_update in range(0, N_update+1, showing_period):
            fid, inception_mean, inception_std, acceptance = calc_scores(G, D, data, evmodel, C, n_update, init_data=init_data)
            fileobj.write("n_update:{}\n".format(n_update))
            fileobj.write("acceptance rate:{}\n".format(acceptance))
            fileobj.write("IS:{}pm{}\n".format(inception_mean, inception_std))
            fileobj.write("FID:{}\n\n".format(fid))

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("scores"):
        os.mkdir("scores")
    evmodel = Inception()
    serializers.load_hdf5('metric/inception_score.model', evmodel)
    G, D, data = load_GD(args.G, args.D)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        evmodel.to_gpu()
        G.to_gpu()
        D.to_gpu()
    G, D = DOT.thermalize_spectral_norm(G, D)
    C = None
    if args.calib=='True':
        C = MH.Calibrator(G, D, fitting_batchsize=1000, data=data)
    main(args, G, D, data, evmodel, C, N_update=args.N_update, showing_period=args.showing_period, init_data=(args.initdata=='True'))
