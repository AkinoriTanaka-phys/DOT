import argparse
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
from model import *
import DOT 

home = "xxxx/codes"
# type the path of the directory

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--evmodel', type=str, default='metric/inception_score.model')
    parser.add_argument('--model', type=str, default='SAGAN')
    parser.add_argument('--transport', type=str, default='dot')
    parser.add_argument('--N_update', type=int, default=100)
    parser.add_argument('--showing_period', type=int, default=10)
    parser.add_argument('--lr', type=float, default=10**(-2)) 
    return parser.parse_args()

## https://github.com/pfnet-research/chainer-gan-lib/blob/master/common/evaluation.py
def get_mean_cov(evmodel, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = evmodel.xp
    ys = xp.empty((n, 2048), dtype=xp.float32)

    for i in range(n_batches):
        #print('Running batch', i + 1, '/', n_batches, '...')
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
            y = evmodel(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = chainer.cuda.to_cpu(xp.mean(ys, axis=0))
    # cov = F.cross_covariance(ys, ys, reduce="no").data.get()
    cov = np.cov(chainer.cuda.to_cpu(ys).T)

    return mean, cov

def FID(m0,c0,m1,c1):
    ret = 0
    ret += np.sum((m0-m1)**2)
    ret += np.trace(c0 + c1 - 2.0*scipy.linalg.sqrtm(np.dot(c0, c1)))
    #ret += np.trace(c0 + c1 - 2.0*np.sqrt(np.dot(c0, c1)+0j))
    return np.real(ret)

def calc_FID(img, evmodel):#, stat_file="%s/cifar-10-fid.npz"%os.path.dirname(__file__)):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    data_m = np.load("{}/metric/cifar_inception_mean.npy".format(home))
    data_c = np.load("{}/metric/cifar_inception_cov.npy".format(home))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(evmodel, img)
    fid = FID(data_m, data_c, mean, cov)
    return fid

###
def calc_scores(G, D, evmodel, transport, N_update, batchsize=100, n_img=50000, k=None, lr=0.1):
    """ dot_mode = [target, latent, bare] """
    for i in range(0, n_img, batchsize):
        im = DOT.make_image(G, D, batchsize, N_update=N_update, ot=True, mode=transport, k=k, lr=lr)
        im = np.asarray(np.clip(im * 127.5 + 127.5, 0.0, 255.0), dtype=np.float32)
        if i==0:
            ims = im
        else:
            ims = np.concatenate((ims, im))

    if args.samples > 0:
        ims = ims[:args.samples]

    fid = calc_FID(ims, evmodel)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(evmodel, ims)
    return fid, mean, std

def load_GD(model):
    try:
        if model=='SAGAN':
            G = SADCGANGenerator()
            D = SADCGANDiscriminator()
        else:
            raise Exception
    except Exception:
        print("Only SAGAN available, sorry.")        
    serializers.load_npz("{}/trained_models/G_cifar_{}.npz".format(home, model), G)
    serializers.load_npz("{}/trained_models/D_cifar_{}.npz".format(home, model), D)
    return G, D



def main(args, G, D, evmodel, k, transport, N_update, showing_period):
    lr = float(args.lr)
    filename="{}_{}_Nupdate{}_lr{}_k{}_{}.txt".format('cifar', args.model, N_update, args.lr, k, transport)
    with open("{}/scores/{}".format(home, filename), "w") as fileobj:
        for n_update in range(0, N_update+1, showing_period):
            fid, inception_mean, inception_std = calc_scores(G, D, evmodel, transport, n_update, k=k, lr=args.lr)
            fileobj.write("n_update:{}\n".format(n_update))
            fileobj.write("k:{}\n".format(cuda.to_cpu(k)))
            fileobj.write("IS:{}pm{}\n".format(inception_mean, inception_std))
            fileobj.write("FID:{}\n\n".format(fid))

if __name__ == '__main__':
    args = parse_args()
    evmodel = Inception()
    serializers.load_hdf5('{}/metric/inception_score.model'.format(home), evmodel)
    G, D = load_GD(args.model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        evmodel.to_gpu()
        G.to_gpu()
        D.to_gpu()
    G, D = DOT.thermalize_spectral_norm(G, D)
    k = DOT.eff_k(G, D)
    main(args, G, D, evmodel, k, transport=args.transport, N_update=args.N_update, showing_period=args.showing_period)
