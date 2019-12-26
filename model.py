import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import cupy as xp

from snd.sn_linear import SNLinear
from snd.sn_convolution_2d import SNConvolution2D
from snd.sn_deconvolution_2d import SNDeconvolution2D

#from sa.attention import Self_Attn 
# I'm sorry SA is not available on this repository because of problem on LICENCE.
# Our experiment in the paper is based on similar implementation on PyTorch implementation: https://github.com/heykeetae/Self-Attention-GAN

### for 2d
class Generator(chainer.Chain):
    def __init__(self, n_hidden=2, noize='uni', non_linear=None, final=None):
        self.final = final
        self.non_linear = non_linear
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.noize = noize
        with self.init_scope():
            init = chainer.initializers.HeNormal(scale=0.8)
            self.l0 = L.Linear(self.n_hidden, 256, initialW=init)
            self.l1 = L.Linear(None, 256, initialW=init)
            self.l2 = L.Linear(None, 256, initialW=init)
            self.l4 = L.Linear(None, 2, initialW=init)

    def make_hidden(self, batchsize):
        return xp.random.uniform(-1, 1, (batchsize, self.n_hidden)).astype(xp.float32)

    def __call__(self, z, train=True):
        h = self.non_linear(self.l0(z))
        h1 = self.non_linear(self.l1(h))
        h = self.non_linear(self.l2(h1))
        h = self.final(self.l4(h))
        h = F.reshape(h, (len(z), 2))
        return h

    def sampling(self, batchsize):
        z = self.make_hidden(batchsize)
        return self(z)

class Discriminator(chainer.Chain):
    def __init__(self, non_linear=None, final=None):
        self.non_linear = non_linear
        self.final = final
        super(Discriminator, self).__init__()
        with self.init_scope():
            init = chainer.initializers.HeNormal(scale=0.8)
            self.l1 = L.Linear(2, 512, initialW=init)
            self.l2 = L.Linear(None, 512, initialW=init)
            self.l3 = L.Linear(None, 512, initialW=init)
            self.l4 = L.Linear(None, 1, initialW=init)

    def __call__(self, x):
        h = self.non_linear(self.l1(x))
        h = self.non_linear(self.l2(h))
        h = self.non_linear(self.l3(h))
        h = self.final(self.l4(h))
        return h

#### for images
#### https://github.com/pfnet-research/chainer-gan-lib/blob/master/common/net.py
# Copyright (c) 2017 pfnet-research
# Released under the MIT license
# https://github.com/pfnet-research/chainer-gan-lib/blob/master/LICENSE
class DCGANGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=512, wscale=0.02,
                 z_distribution="uniform", hidden_activation=F.relu, output_activation=F.tanh, use_bn=True):
        super(DCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return xp.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(xp.float32)
        elif self.z_distribution == "uniform":
            return xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x

    def sampling(self, batchsize):
        z = self.make_hidden(batchsize)
        return self(z)

class SADCGANGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=512, wscale=0.02,
                 z_distribution="uniform", hidden_activation=F.relu, output_activation=F.tanh, use_bn=True):
        super(SADCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = SNLinear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = SNDeconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = SNDeconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            #self.sa1 = Self_Attn(ch//4)
            self.dc3 = SNDeconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = SNDeconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return xp.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(xp.float32)
        elif self.z_distribution == "uniform":
            return xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            #h,_ = self.sa1(h)
            h = self.hidden_activation(self.dc3(h))
            h = self.dc4(h)
            x = self.output_activation(h)
        else:
            h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            #h, _ = self.sa1(h)
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            h = self.dc4(h)
            x = self.output_activation(h)
        return x

    def sampling(self, batchsize):
        z = self.make_hidden(batchsize)
        return self(z)

class WGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(WGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h0 = F.leaky_relu(self.c0(self.x))
        self.h1 = F.leaky_relu(self.c1(self.h0))
        self.h2 = F.leaky_relu(self.c1_0(self.h1))
        self.h3 = F.leaky_relu(self.c2(self.h2))
        self.h4 = F.leaky_relu(self.c2_0(self.h3))
        self.h5 = F.leaky_relu(self.c3(self.h4))
        self.h6 = F.leaky_relu(self.c3_0(self.h5))
        return self.l4(self.h6)

class SNDCGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SNDCGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = SNConvolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = SNConvolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = SNConvolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = SNConvolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = SNConvolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = SNConvolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = SNConvolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c2_1(h))
        h = F.leaky_relu(self.c3_0(h))
        return self.l4(h)

class SADCGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SADCGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = SNConvolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = SNConvolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            #self.sa1 = Self_Attn(ch//4)
            self.c1_0 = SNConvolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = SNConvolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = SNConvolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = SNConvolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = SNConvolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        #h,_ = self.sa1(h)
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c2_1(h))
        h = F.leaky_relu(self.c3_0(h))
        return self.l4(h)

### 
class UpResBlock(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, in_ch, out_ch=None, wscale=0.02):
        if out_ch==None:
            self.out_ch=in_ch
        else:
            self.out_ch=out_ch
        self.in_ch = in_ch
        super(UpResBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(self.in_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(self.out_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(self.in_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(self.in_ch)
            self.bn1 = L.BatchNormalization(self.out_ch)

    def __call__(self, x):
        h = self.c0(F.unpooling_2d(F.relu(self.bn0(x)), 2, 2, 0, cover_all=False))
        h = self.c1(F.relu(self.bn1(h)))
        hs = self.cs(F.unpooling_2d(x, 2, 2, 0, cover_all=False))
        return h + hs

class SNUpResBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch=None, wscale=0.02):
        if out_ch==None:
            self.out_ch=in_ch
        else:
            self.out_ch=out_ch
        self.in_ch = in_ch
        super(SNUpResBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = SNConvolution2D(self.in_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(self.out_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.cs = SNConvolution2D(self.in_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(self.in_ch)
            self.bn1 = L.BatchNormalization(self.out_ch)

    def __call__(self, x):
        h = self.c0(F.unpooling_2d(F.relu(self.bn0(x)), 2, 2, 0, cover_all=False))
        h = self.c1(F.relu(self.bn1(h)))
        hs = self.cs(F.unpooling_2d(x, 2, 2, 0, cover_all=False))
        return h + hs

class ResnetGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, z_distribution="normal", wscale=0.02):
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        super(ResnetGenerator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(n_hidden, n_hidden * bottom_width * bottom_width)
            self.r0 = UpResBlock(n_hidden)
            self.r1 = UpResBlock(n_hidden)
            self.r2 = UpResBlock(n_hidden)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.c3 = L.Convolution2D(n_hidden, 3, 3, 1, 1, initialW=w)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return xp.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(xp.float32)
        elif self.z_distribution == "uniform":
            return xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, x):
        h = F.reshape(F.relu(self.l0(x)), (x.shape[0], self.n_hidden, self.bottom_width, self.bottom_width))
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        h = self.bn2(F.relu(h))
        h = F.tanh(self.c3(h))
        return h

    def sampling(self, batchsize):
        z = self.make_hidden(batchsize)
        return self(z)

class SAResnetGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, z_distribution="normal", wscale=0.02):
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        super(SAResnetGenerator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = SNLinear(n_hidden, n_hidden * bottom_width * bottom_width)
            self.r0 = SNUpResBlock(n_hidden)
            self.r1 = SNUpResBlock(n_hidden)
            self.r2 = SNUpResBlock(n_hidden)
            #self.sa1 = Self_Attn(n_hidden)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.c3 = SNConvolution2D(n_hidden, 3, 3, 1, 1, initialW=w)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return xp.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(xp.float32)
        elif self.z_distribution == "uniform":
            return xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, x):
        h = F.reshape(F.relu(self.l0(x)), (x.shape[0], self.n_hidden, self.bottom_width, self.bottom_width))
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        #h,_ = self.sa1(h)
        h = self.bn2(F.relu(h))
        h = F.tanh(self.c3(h))
        return h

    def sampling(self, batchsize):
        z = self.make_hidden(batchsize)
        return self(z)


class DownResBlock1(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(3, ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0((self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4


class SNDownResBlock1(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(SNDownResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cs = SNConvolution2D(3, ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0((self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4


class DownResBlock2(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, in_ch, out_ch=None):
        if out_ch==None:
            self.out_ch = in_ch
        else:
            self.out_ch = out_ch
        self.in_ch = in_ch
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(self.in_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(self.out_ch, self.out_ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(self.in_ch, self.out_ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4

class SNDownResBlock2(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, in_ch, out_ch=None):
        if out_ch==None:
            self.out_ch = in_ch
        else:
            self.out_ch = out_ch
        self.in_ch = in_ch
        w = chainer.initializers.Normal(0.02)
        super(SNDownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(self.in_ch, self.out_ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(self.out_ch, self.out_ch, 4, 2, 1, initialW=w)
            self.cs = SNConvolution2D(self.in_ch, self.out_ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4

class DownResBlock3(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, in_ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        return self.h4

class SNDownResBlock3(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(SNDownResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        return self.h4

class ResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(ResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = DownResBlock1(128)
            self.r1 = DownResBlock2(128)
            self.r2 = DownResBlock3(128)
            self.r3 = DownResBlock3(128)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        h = self.r0(x)
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = F.relu(h)
        h = self.l4(h)
        return h

class SNResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SNResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = SNDownResBlock1(128)
            self.r1 = SNDownResBlock2(128)
            self.r2 = SNDownResBlock3(128)
            self.r3 = SNDownResBlock3(128)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        return self.l4(F.relu(self.h4))

class SAResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SAResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = SNDownResBlock1(128)
            self.r1 = SNDownResBlock2(128)
            self.r2 = SNDownResBlock3(128)
            self.r3 = SNDownResBlock3(128)
            #self.sa1 = Self_Attn(128)
            self.l4 = SNLinear(None, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        #self.h5,_ = self.sa1(self.h4) 
        self.h5 = self.h4
        return self.l4(F.relu(self.h5))
