import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from source.miscs.random_samples import sample_categorical, sample_continuous


import numpy as np
from chainer import cuda
#import numpy as xp
gpu_device = 0
cuda.get_device(gpu_device).use()
import cupy as xp


def l2_norm(x):
    return F.sqrt(F.batch_l2_norm_squared(x))

distance=l2_norm

def eff_k_cond(G, D, trial=100, label=None):
    z1 = sample_continuous(128, trial, distribution=G.distribution, xp=xp)
    z2 = sample_continuous(128, trial, distribution=G.distribution, xp=xp)
    
    labels = label*xp.ones(trial).astype(xp.int32)
    with chainer.using_config('train', False):
        f1 = D(G(batchsize=trial, y=labels, z=z1))
        f2 = D(G(batchsize=trial, y=labels, z=z2))
            
    nu =  distance(f2 - f1)
    de_l = distance(z2 - z1)

    return cuda.to_cpu(F.max(nu/de_l).data)

def return_ks(G, D, trial=50, nlabels=1000):
    ks = []
    for y in range(nlabels):
        ks.append(eff_k_cond(G, D, trial=trial, label=y))
    
    return np.array(ks).reshape(nlabels, 1)


###
class Transporter_in_latent():
    def __init__(self, G, D, k, opt, zy_xp, labels, mode):
        self.G = G
        self.D = D
        self.opt = opt
        self.zy = cuda.to_gpu(zy_xp)
        self.labels = cuda.to_gpu(labels)
        self.mode = mode 
        self.onegrads = xp.ones(zy_xp.shape[0]).reshape(zy_xp.shape[0], 1).astype(xp.float32)
        self.lc = k
        self.dist = G.distribution
        
    def get_z_va(self):
        return self.z.W
    
    def set_(self, zy_xp):
        self.z = L.Parameter(zy_xp)
        self.z.to_gpu
        self.opt.setup(self.z)
    
    def H_zy(self, z):
        x = self.G(batchsize=z.shape[0], y=self.labels, z=z)
        with chainer.using_config('train', False):
            d = self.D(x, y=self.labels)
            if self.mode=='target':
                y = self.G(batchsize=z.shape[0], y=self.labels, z=self.zy)
                obj = - d/self.lc + F.reshape(distance(x - y+0.001), d.shape)
                return obj#- self.D(x)/self.lc + F.reshape(distance(x - y), self.D(x).shape) 
            elif self.mode=='latent':
                #print("z", F.identity(z))
                #print("self.zy", self.zy)
                obj = - d/self.lc + F.reshape(distance(z - Variable(self.zy)+0.001), d.shape)
                #print()
                return obj#F.reshape(distance(z - self.zy), self.D(x).shape) 
            else:
                return - d/self.lc
        
    def step(self):
        z = self.get_z_va()
        z.cleargrad()
        loss = self.H_zy(z)
        loss.grad = self.onegrads
        loss.backward()
        if self.dist=='uniform':
            #print(self.opt.target.W.data.shape)
            #print(self.opt.lr)
            #self.opt.target.W.data -= self.opt.lr * xp.sign(z.grad)
            #self.opt.target.W.data = xp.clip(self.opt.target.W.data, -1, 1)
            self.opt.update()
            self.opt.target.W.data = xp.clip(self.opt.target.W.data, -1, 1)

        elif self.dist=='normal':
            #print('zshape', z.shape)
            bs, dim  = z.shape
            #print('zshape', z.shape)
            #print('zgrad shape', z.grad.shape)
            prod = F.sum(F.batch_matmul(z.grad.reshape(bs, dim), z.data.reshape(bs, dim), transa=True), 1).reshape(bs, 1) 
            #F.sum(F.batch_matmul(z.grad, z.data, transa=True), 1).reshape(bs, 1, 1)
            #print("zgrad type", type(z.grad))
            #print("z type", type(z.data))
            #print("prod type", type(prod))
            z.grad = z.grad - z.data*(prod.data)/11.31
            self.opt.update()
            #self.opt.target.W.data = xp.clip(self.opt.target.W.data, -1, 1)
        
###
def thermalize_spectral_norm(G, D):
    for i in range(100): # thermalize G, D
        with chainer.using_config('train', False):
            x = G(batchsize=10, y=None, z=None)
            d = D(x, y=None)
    return G, D
        
def discriminator_optimal_transport_from(y_or_z_xp, transporter, N_update=10):
    transporter.set_(y_or_z_xp)
    #print(y_or_z_xp == transporter.get_z_va().data)
    for i in range(N_update):            
        transporter.step()

def make_image(G, D, batchsize, N_update=100, ot=True, mode='latent', k=1, lr=0.05, optmode='sgd'):
    #z = G.make_hidden(batchsize)
    label = sample_categorical(1000, batchsize, distribution="uniform", xp=xp)
    labels = label*xp.ones(batchsize).astype(xp.int32)
    zs = sample_continuous(128, batchsize, distribution=G.distribution, xp=xp)
    if k != 1:
        k = k(labels).data

    with chainer.using_config('train', False):
        if ot:
            z_xp = zs
            #print(type(z_xp))
            if optmode=='sgd':
                Opt = chainer.optimizers.SGD(lr)#, beta1=0.0, beta2=0.9)
            elif optmode=='adam':
                Opt = chainer.optimizers.Adam(lr, beta1=0.0, beta2=0.9)
            T = Transporter_in_latent(G, D, k, Opt, z_xp, labels, mode=mode)#, sign=True)
            discriminator_optimal_transport_from(z_xp, T, N_update)
            tz_y = T.get_z_va().data
            #print(type(tz_y))
            y = G(batchsize=batchsize, y=labels, z=tz_y)
        else:
            y = G(batchsize=batchsize, y=labels, z=zs)
    return cuda.to_cpu(y.data) 
