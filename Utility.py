
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-06-16 17:01:55
# @modify date 2020-06-16 17:01:55

# system
import glob
import os
import time
import multiprocessing
from tqdm import tqdm

# numeric data processing
import numpy as np
import pandas as pd
import awkward

# root io
import ROOT as root
from ROOT import TLorentzVector
import uproot

# cuda
import pycuda as cu
import pycuda.autoinit
import pycuda.gpuarray
import pycuda.scan
import pycuda.elementwise
import pycuda.compiler
import pycuda.driver as cuda


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

sign = lambda a: (a>0) - (a<0)




knl_exsumscan = cu.scan.ExclusiveScanKernel(np.int32, "a+b", neutral=0)

knl_compact = cu.compiler.SourceModule("""
    __global__ void compact(int *compaction, int *exsumscan, int *mask, const int n) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i<n) if(mask[i]==1) compaction[exsumscan[i]] = i;
        }
    """).get_function("compact")

def parallelCompact(mask):
    nin = len(mask)
    # get exclusive sum scan
    exsumscan = knl_exsumscan(mask.copy())
    # get compaction
    nout = int(cu.gpuarray.sum(mask).get()) 
    compaction = cu.gpuarray.empty(nout, np.int32)
    n = cu.gpuarray.to_gpu(np.array(nin,dtype=np.int32))
    knl_compact(compaction, exsumscan, mask, n, grid  = (int(nin/1024)+1,1,1), block=(1024,1,1))
    return compaction