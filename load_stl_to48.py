import argparse
import numpy as np
import chainer
from chainer import cuda
from chainer import Variable
import chainer.functions as F
import cupy as xp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stlpath', type=str, default="training_data/STL96.npy")
    return parser.parse_args()

def return_size48_array_cpu(stl_path):
    batchsize = 1000
    stl_96_cpu = np.load(stl_path).astype(np.float32)
    #print(stl_96_cpu.shape)
    for n in range(0, 100000//batchsize):
        print(n*batchsize, (n+1)*batchsize)
        stl_96_gpu = Variable(cuda.to_gpu(stl_96_cpu[n*batchsize:(n+1)*batchsize]))
        #print(stl_96_gpu.shape)
        x = F.average_pooling_2d(stl_96_gpu, 2).data
        if n==0:
            stl_48_cpu = cuda.to_cpu(x)
        else:
            stl_48_cpu = np.concatenate([stl_48_cpu, cuda.to_cpu(x)], axis=0)
    return stl_48_cpu

if __name__ == '__main__':
    args = parse_args()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    stl_48 = return_size48_array_cpu(args.stlpath)
    print("saved shape:", stl_48.shape)
    np.save("training_data/STL48.npy", stl_48)

