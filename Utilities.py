
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


dtype2callable = {
    np.dtype('bool'): np.bool, 
    np.dtype('uint32'): np.uint32, 
    np.dtype('int32'): np.int32, 
    np.dtype('float32'): np.float32
    }