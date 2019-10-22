import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from snd.sn_convolution_2d import SNConvolution2D


class Self_Attn(chainer.Chain):
    def __init__(self, in_dim):#, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation
        with self.init_scope():
            self.query_conv = SNConvolution2D(in_channels = in_dim, out_channels = in_dim//8, ksize=1, \
                                          stride=1, pad=0, nobias=False)
            self.key_conv = SNConvolution2D(in_channels = in_dim, out_channels = in_dim//8, ksize=1, \
                                          stride=1, pad=0, nobias=False)
            self.value_conv = SNConvolution2D(in_channels = in_dim, out_channels = in_dim, ksize=1, \
                                          stride=1, pad=0, nobias=False)
            self.gamma = chainer.Parameter(initializer=chainer.initializers.Zero(), shape=(1,))

        #self.softmax  = F.softmax(axis=-1) #
    def __call__(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.shape
        #print(self.query_conv(x).shape)
        proj_query  = F.transpose(F.reshape(self.query_conv(x), (m_batchsize, -1, width*height)),\
                                  (0,2,1))
        #print(proj_query.shape) # B x C x N
        proj_key =  F.reshape(self.key_conv(x), (m_batchsize, -1, width*height))
        #print(proj_key.shape) # B x C x N
        energy =  F.matmul(proj_query, proj_key)
        #torch.bmm(proj_query, proj_key) # transpose check
        attention = F.softmax(energy, axis=2) # B x N x N 
        proj_value = F.reshape(self.value_conv(x), (m_batchsize, -1, width*height)) # B X C X N

        out = F.matmul(proj_value, F.transpose(attention, (0,2,1)))
        out = F.reshape(out, (m_batchsize,C,width,height))
        #print(self.gamma)
        
        out = F.broadcast_to(self.gamma, out.shape)*out + x
        return out, attention
