import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer import cuda
import cupy as xp

from sklearn.linear_model import LogisticRegression
import numpy as np
        
def Z(C, G, D, x_real):
    bs = x_real.shape[0]
    with chainer.using_config('train', False):
        x_fake = G.sampling(bs)
    y_real = np.ones(bs)
    y_fake = np.zeros(bs)
    d_real = C(cuda.to_cpu(D(x_real).data))
    d_fake = C(cuda.to_cpu(D(x_fake).data))
    num = np.sum(y_real - d_real + y_fake - d_fake)
    den = np.sum(np.sqrt(d_real*(1- d_real)) + np.sqrt(d_fake*(1- d_fake)))
    return num/den        

class Calibrator():
    def __init__(self, G, D, fitting_batchsize=1000, data=None):
        self.clf = LogisticRegression()
        datapath='training_data/{}.npy'.format(data)
        X_train = (xp.load(datapath).astype(xp.float32))*2 - 1
        x_0 = cuda.to_cpu(D(G.sampling(fitting_batchsize)).reshape(fitting_batchsize, 1).data)
        y_0 = np.zeros(len(x_0))
        x_1 = cuda.to_cpu(D(X_train[:fitting_batchsize]).reshape(fitting_batchsize, 1).data)
        y_1 = np.ones(len(x_1))
        X = np.concatenate([x_0, x_1])
        Y = np.concatenate([y_0, y_1])
        self.clf.fit(X, Y)
        self.Zvalue = Z(self, G, D, X_train[10000:10000+1000])
        assert(self.Zvalue < 2 or self.Zvalue > -2)

    def __call__(self, d):
        return self.clf.predict_proba(d)[:, 1]

def make_image2(G, D, batchsize, C=None, N_update=640, initial=None):
    try:
        initial_exists = initial.any()
    except AttributeError:
        initial_exists = False
    acceptance_rate_seq = []
    accepted_num = 0
    counter = 0
    if N_update==0:
        with chainer.using_config('train', False):
           x = cuda.to_cpu(G.sampling(batchsize).data)
           all_accept = (np.ones(batchsize)==np.ones(batchsize)).reshape(batchsize, 1, 1, 1)
    while (accepted_num<batchsize):
        for k in range(N_update):
            if k==0:
                if initial_exists:
                    x = cuda.to_cpu(initial)
                else:
                    with chainer.using_config('train', False):
                        x = cuda.to_cpu(G.sampling(batchsize).data) 
            with chainer.using_config('train', False):
                x_gpu = cuda.to_gpu(x)
                xprime_gpu = G.sampling(batchsize).data
                y = cuda.to_cpu(D(x_gpu).data)
                yprime = cuda.to_cpu(D(xprime_gpu).data)
            if C!=None:
                y = C(y.reshape(batchsize, 1))
                yprime = C(yprime.reshape(batchsize, 1))
                alpha = (y**(-1) - 1)/(yprime**(-1) - 1)
            else:
                y = F.sigmoid(y.reshape(batchsize, 1))
                yprime = F.sigmoid(yprime.reshape(batchsize, 1))
                alpha = ((y**(-1) - 1)/(yprime**(-1) - 1)).data.reshape(batchsize)
            U = np.random.uniform(0,1, batchsize)
            accepted = (U <= alpha).reshape(batchsize, 1, 1, 1)
            x = (accepted*cuda.to_cpu(xprime_gpu) + (1-accepted)*cuda.to_cpu(x_gpu)).astype(np.float32)
            acceptance_rate_seq.append(np.mean(accepted))
            if k==0:
                all_accept = accepted
            else:
                all_accept = all_accept + accepted
        all_accept = (all_accept).reshape(-1)
        if initial_exists:
            if counter==0:
                x_accepted = x[all_accept]
                accepted_num = x_accepted.shape[0]
            else:
                x_accepted = np.append(x_accepted, x[all_accept], axis=0)
                accepted_num = x_accepted.shape[0]
            counter+=1
        else:
            x_accepted = x[all_accept]
            break
    x_accepted = x_accepted[:batchsize]
    return x_accepted, np.mean(acceptance_rate_seq)
