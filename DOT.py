import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer import cuda
import cupy as xp

def l2_norm(x):
    return F.sqrt(F.batch_l2_norm_squared(x))

distance=l2_norm

def eff_k(G, D, trial=100):
    with chainer.using_config('train', False):
        z1 = cuda.to_gpu(G.make_hidden(trial))
        x1 = G(z1)
        f1 = D(x1)
        z2 = cuda.to_gpu(G.make_hidden(trial))
        x2 = G(z2)
        f2 = D(x2)
        nu =  distance(f2 - f1)
        de_l = distance(z2 - z1)
        return F.max(nu/de_l).data

def eff_K(G, D, trial=100):
    with chainer.using_config('train', False):
        z1 = cuda.to_gpu(G.make_hidden(trial))
        x1 = G(z1)
        f1 = D(x1)
        z2 = cuda.to_gpu(G.make_hidden(trial))
        x2 = G(z2)
        f2 = D(x2)
        nu =  distance(f2 - f1)
        de_L = distance(x2 - x1)
        return F.max(nu/de_L).data

class Transporter_in_target():
    def __init__(self, G, D, K, opt, y_xp, mode):
        self.G = G
        self.D = D
        self.opt = opt
        self.y = y_xp.copy()
        self.mode = mode 
        self.onegrads = xp.ones(y_xp.shape[0]).reshape(y_xp.shape[0], 1).astype(xp.float32)
        self.lc = K
    
    def get_x_va(self):
        return self.x.W
    
    def set_(self, y_xp):
        self.x = L.Parameter(y_xp.copy())
        self.opt.setup(self.x)
    
    def H_y(self, x):
        if self.mode=='dot':
            return - self.D(x)/self.lc + F.reshape(distance(x - self.y + 0.001),self.D(x).shape) 
        else:
            return - self.D(x)/self.lc
        
    def step(self):
        x = self.get_x_va()
        x.cleargrad()
        loss = self.H_y(x)
        loss.grad = self.onegrads
        loss.backward()
        self.opt.update()
        self.opt.target.W.data = xp.clip(self.opt.target.W.data, -2, 2)
        
class Transporter_in_latent():
    def __init__(self, G, D, k, opt, zy_xp, mode):
        self.G = G
        self.D = D
        self.opt = opt
        self.zy = cuda.to_gpu(zy_xp)
        self.mode = mode 
        self.onegrads = xp.ones(zy_xp.shape[0]).reshape(zy_xp.shape[0], 1).astype(xp.float32)
        self.lc = k
        self.dist = G.z_distribution
        
    def get_z_va(self):
        return self.z.W
    
    def set_(self, zy_xp):
        self.z = L.Parameter(zy_xp)
        self.z.to_gpu
        self.opt.setup(self.z)
    
    def H_zy(self, z):
        x = self.G(z)
        with chainer.using_config('train', False):
            if self.mode=='dot':
                return - self.D(x)/self.lc + F.reshape(distance(z - Variable(self.zy) + 0.001), self.D(x).shape) 
            else:
                return - self.D(x)/self.lc
        
    def step(self):
        z = self.get_z_va()
        z.cleargrad()
        loss = self.H_zy(z)
        loss.grad = self.onegrads
        loss.backward()
        if self.dist=='uniform':
            self.opt.update()
            self.opt.target.W.data = xp.clip(self.opt.target.W.data, -1, 1)

        elif self.dist=='normal':
            bs, dim, _, _  = z.shape
            prod = F.sum(F.batch_matmul(z.grad.reshape(bs, dim), z.data.reshape(bs, dim), transa=True), 1).reshape(bs, 1, 1, 1) 
            z.grad = z.grad - z.data*(prod.data)/11.31 # 11.31 = sqrt(128)
            self.opt.update()
        
def thermalize_spectral_norm(G, D):
    for i in range(100):
        with chainer.using_config('train', False):
            z = G.make_hidden(10)
            x = G(z)
            d = D(x)
    return G, D
        
def discriminator_optimal_transport_from(y_or_z_xp, transporter, N_update=10):
    transporter.set_(y_or_z_xp)
    for i in range(N_update):            
        transporter.step()

def make_image(G, D, batchsize, N_update=100, ot=True, mode='dot', k=1, lr=0.05, optmode='sgd'):
    z = G.make_hidden(batchsize)
    with chainer.using_config('train', False):
        if ot:
            z_xp = z
            if optmode=='sgd':
                Opt = chainer.optimizers.SGD(lr)
            elif optmode=='adam':
                Opt = chainer.optimizers.Adam(lr, beta1=0.0, beta2=0.9)
            T = Transporter_in_latent(G, D, k, Opt, z_xp, mode=mode)
            discriminator_optimal_transport_from(z_xp, T, N_update)
            tz_y = T.get_z_va().data
            y = G(tz_y)
        else:
            y = G(z)
    return cuda.to_cpu(y.data) 
