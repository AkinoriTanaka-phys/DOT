import argparse
import numpy as np
import scipy
import re
import os
from datetime import datetime

import chainer
from chainer import cuda
from chainer import datasets
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from inception_score import Inception
from inception_score import inception_score

from evaluation import *

import math

import cupy as xp
from model import *
import DOT 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--evmodel', type=str, default='metric/inception_score.model')
    parser.add_argument('--G', type=str, default='trained_models/xxxx')
    parser.add_argument('--D', type=str, default='trained_models/xxxx')
    parser.add_argument('--transport', type=str, default='dot')
    parser.add_argument('--optmode', type=str, default='sgd')
    parser.add_argument('--N_update', type=int, default=100)
    parser.add_argument('--showing_period', type=int, default=10)
    parser.add_argument('--lr', type=float, default=10**(-2))
    parser.add_argument('--k', type=int, default=None)
    return parser.parse_args()

###
def returnG(bw, n, mode, net):
    if mode=='WGAN-GP':
        if net=='DCGAN':
            G = DCGANGenerator(bottom_width=bw)
        elif net=='Resnet':
            G = ResnetGenerator(bottom_width=bw, n_hidden=128*n)
    elif mode=='SNGAN':
        if net=='DCGAN':
            G = DCGANGenerator(bottom_width=bw)
        elif net=='Resnet':
            G = ResnetGenerator(bottom_width=bw, n_hidden=128*n)
    elif mode=='SAGAN':
        if net=='DCGAN':
            G = SADCGANGenerator(bottom_width=bw)
        elif net=='Resnet':
            G = SAResnetGenerator(bottom_width=bw, n_hidden=128*n)
    return G

def returnD(bw, mode, net):
    if mode=='WGAN-GP':
        if net=='DCGAN':
            D = WGANDiscriminator(bottom_width=bw)
        elif net=='Resnet':
            D = ResnetDiscriminator(bottom_width=bw)
    elif mode=='SNGAN':
        if net=='DCGAN':
            D = SNDCGANDiscriminator(bottom_width=bw)
        elif net=='Resnet':
            D = SNResnetDiscriminator(bottom_width=bw)
    elif mode=='SAGAN':
        if net=='DCGAN':
            D = SADCGANDiscriminator(bottom_width=bw)
        elif net=='Resnet':
            D = SAResnetDiscriminator(bottom_width=bw)
    return D

def calc_scores(G, D, data, evmodel, transport, N_update, batchsize=100, n_img=50000, k=None, lr=0.1, optmode='sgd'):
    for i in range(0, n_img, batchsize):
        im = DOT.make_image(G, D, batchsize, N_update=N_update, ot=True, mode=transport, k=k, lr=lr, optmode=optmode)
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

def load_GD(Gfilename, Dfilename):
    Gnet, _, Gdata, Gmode, _, _ = re.split(r'[_]', Gfilename)
    Dnet, _, Ddata, Dmode, _, _ = re.split(r'[_]', Dfilename)
    try:
        if Gdata==Ddata:
            pass
        else:
            raise Exception
    except Exception:
        print("Domain(CIFAR/STL48) should be same in G and D.")

    if Gdata=='STL48':
        bw = 6
        n=1
    elif Gdata=='CIFAR':
        bw = 4
        n=2
    G = returnG(bw, n, Gmode, Gnet)
    D = returnD(bw, Dmode, Dnet)

    serializers.load_npz("trained_models/{}".format(Gfilename), G)
    serializers.load_npz("trained_models/{}".format(Dfilename), D)
    return G, D, Gdata


def main(args, G, D, data, evmodel, k, transport, N_update, showing_period):
    lr = float(args.lr)
    filename="uncond_" + datetime.now().strftime("%Y_%m_%d_%H%M%S")+".txt"
    with open("scores/{}".format(filename), "w") as fileobj:
        fileobj.write("{}\n".format(args.G))
        fileobj.write("{}\n".format(args.D))
        fileobj.write("DOTmode:{}\n".format(transport))
        fileobj.write("lr:{}\n".format(args.lr))
        fileobj.write("optimizer:{}\n".format(args.optmode))
        fileobj.write("k:{}\n\n".format(cuda.to_cpu(k)))
        for n_update in range(0, N_update+1, showing_period):
            fid, inception_mean, inception_std = calc_scores(G, D, data, evmodel, transport, n_update, k=k, lr=args.lr, optmode=args.optmode)
            fileobj.write("n_update:{}\n".format(n_update))
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
    if args.k==None:
        k = DOT.eff_k(G, D)
    else:
        k = args.k*xp.ones([1])
    main(args, G, D, data, evmodel, k, transport=args.transport, N_update=args.N_update, showing_period=args.showing_period)
