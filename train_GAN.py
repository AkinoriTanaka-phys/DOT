import argparse

import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer import cuda
#import numpy as xp
import cupy as xp
from chainer.training import extensions

from evaluation import *

#import ot
import os
import re

from model import *

# basic functions
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', type=str, default='CIFAR')
    parser.add_argument('--mode', type=str, default='WGAN-GP/SNGAN/SAGAN')
    parser.add_argument('--net', type=str, default='DCGAN/Resnet')
    parser.add_argument('--objective', type=str, default='Wasserstein/NonSaturating/Hinge')
    parser.add_argument('--iters', type=int, default=300000)
    parser.add_argument('--report', type=int, default=10000)
    return parser.parse_args()

def samples_interpolated_from_false_samples_to(x, G=None):
    true_samples = Variable(x)
    batchsize = x.shape[0]
    z = G.make_hidden(batchsize)
    false_samples = Variable(G(z).data)
    #if image
    e = xp.random.uniform(0., 1., (batchsize, 1, 1, 1))
    x_hat = e * true_samples + (1 - e) * false_samples
    return x_hat

def gradient_penalty(G=None, D=None, x=None, epsilon=None):
    x_hat = samples_interpolated_from_false_samples_to(x, G=G)
    grad, = chainer.grad([D(x_hat)], [x_hat], enable_double_backprop=True)
    grad = F.sqrt(F.batch_l2_norm_squared(grad))
    return epsilon*F.mean_squared_error(grad, xp.ones_like(grad.data))

def null_penalty(G=None, D=None, x=None, epsilon=None):
    return 0

# GAN updater
def dloss_wgan(p, n):
    return F.mean(p - n)
    
def gloss_wgan(n):
    return F.mean(-n)

def dloss_softplus(p, n):
    return -F.mean(F.softplus(-p) + F.softplus(n))

def gloss_softplus(n):
    return F.mean(F.softplus(-n))

def dloss_hinge(p, n):
    zero = 0*p
    return F.mean(F.minimum(zero, -1+p) + F.minimum(zero, -1-n))

def gloss_hinge(n):
    zero = 0*n
    return F.mean(-n)

class GAN_Updater(chainer.training.StandardUpdater):
    def __init__(self, iterator=None, epsilon=None, D_iters=None, 
                 opt_g=None, opt_d=None, objective='NonSaturating', penalty=null_penalty,
                 device=None):
        if objective == 'Wasserstein':
            self.gloss = gloss_wgan
            self.dloss = dloss_wgan
        elif objective == 'NonSaturating':
            self.gloss = gloss_softplus
            self.dloss = dloss_softplus
        elif objective == 'Hinge':
            self.gloss = gloss_hinge
            self.dloss = dloss_hinge
        else:
            raise NameError("objective should be in ['Wassertsein', 'NonSaturating', 'Hinge']")

        if isinstance(iterator, chainer.dataset.iterator.Iterator):
            iterator = {'main': iterator}
        
        #self.mode = mode
        
        self.penalty = penalty
        self.epsilon = epsilon
        
        self._iterators = iterator
        self.generator = opt_g.target
        self.discriminator = opt_d.target
        self.D_iters = D_iters
        self._optimizers = {'generator': opt_g, 'discriminator': opt_d}
        self.device = device
        self.converter = chainer.dataset.concat_examples
        self.iteration = 0
        
    def update_core(self):
        # train discriminator
        for t in range(self.D_iters):
            self.discriminator.cleargrads()
            self.generator.cleargrads()
            # read data
            batch = self._iterators['main'].next()
            true_samples = self.converter(batch, self.device)
            batchsize = true_samples.shape[0]
            fake_samples = self.generator.sampling(batchsize)
            
            D_positive = self.discriminator(true_samples)
            D_negative = self.discriminator(fake_samples)
            
            dloss = self.dloss(D_positive, D_negative)
            penalty = self.penalty(G=self.generator, D=self.discriminator, 
                                   x=true_samples, epsilon=self.epsilon)
            dloss_total = -dloss + penalty

            # update discriminator
            dloss_total.backward()
            self._optimizers['discriminator'].update()

        # train generator
        g_iters = 1
        for t in range(g_iters):
            # read data
            self.discriminator.cleargrads()
            self.generator.cleargrads()
        
        # generate and compute loss
            fake_samples = self.generator.sampling(batchsize)
            D_negative = self.discriminator(fake_samples)
            gloss = self.gloss(D_negative)

        # update generator
            gloss.backward()
            self._optimizers['generator'].update()
       
        #inv_dloss = self.dloss(D_negative, D_positive) 
        # report
        #chainer.reporter.report({'dloss': dloss, 'penalty': penalty})#, 'wasserstein': wd})
        #chainer.reporter.report({'inv_dloss': inv_dloss})

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


def get_initialized_GD_optimizers_Diters_penalty(mode='SNGAN', net='DCGAN', data='STL48'):
    TTUR = False

    if data=='STL48':
        bw = 6
        n=1
    elif data=='CIFAR':
        bw = 4 
        n=2
    G = returnG(bw, n, mode, net)
    D = returnD(bw, mode, net)

    if mode=='WGAN-GP':
        penalty = gradient_penalty
        d_iters = 5
    elif mode=='SNGAN':
        penalty = null_penalty
        d_iters = 5
    elif mode=='SAGAN':
        penalty = null_penalty
        d_iters = 1
        TTUR = True

    G.to_gpu()
    D.to_gpu()
    if TTUR:
        opt_g = chainer.optimizers.Adam(0.0001, beta1=0.0, beta2=0.9)
        opt_d = chainer.optimizers.Adam(0.0004, beta1=0.0, beta2=0.9)
    else:
        opt_g = chainer.optimizers.Adam(0.0002, beta1=0.0, beta2=0.9)
        opt_d = chainer.optimizers.Adam(0.0002, beta1=0.0, beta2=0.9)
    opt_g.setup(G)
    opt_d.setup(D)

    return G, D, opt_g, opt_d, d_iters, penalty
        
def main(data='STL48', batchsize=64, e=None, device=0, 
         mode='SNGAN-EP', net='DCGAN', objective='NonSaturating', show=False, plot=False, num_iterations=101, report_period=None):
    datapath = "training_data/{}.npy".format(data)
    global X_train 
    X_train = xp.load(datapath)*2 - 1 # it takes value in [-1, 1]
    X_train_cpu = chainer.cuda.to_cpu(X_train).astype(np.float32)
    X_train = chainer.cuda.to_gpu(X_train_cpu)
    train_iter = chainer.iterators.SerialIterator(X_train, batchsize, True, True)

    G, D, opt_g, opt_d, d_iters, penalty = get_initialized_GD_optimizers_Diters_penalty(mode=mode, net=net, data=data) 
    updater = GAN_Updater(iterator=train_iter, epsilon=e, D_iters=d_iters, 
                 opt_g=opt_g, opt_d=opt_d, objective=objective, penalty=penalty,
                 device=device)
    
    trainer = chainer.training.Trainer(updater, (num_iterations, 'iteration'), out='result/{}_{}_{}_{}'.format(net, mode, data, objective))
    trainer.extend(calc_inception(G, data=data), trigger=(report_period, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(report_period, 'iteration')))
    
    trigger = chainer.training.triggers.MaxValueTrigger('inception_mean', trigger=(report_period, 'iteration'))
    trainer.extend(save_models(G, D, net=net, data=data, mode=mode, objective=objective), trigger=trigger)

    trainer.run()       

if __name__ == "__main__":
    args = parse_args()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use() 
    main(data=args.data, batchsize=64, e=10.0, device=args.gpu,
         mode=args.mode, net=args.net, objective=args.objective, show=False, plot=False, num_iterations=args.iters, report_period=args.report)
