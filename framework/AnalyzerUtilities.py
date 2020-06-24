
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-06-16 17:01:55
# @modify date 2020-06-16 17:01:55

MAXNLEPTON = 4


######################################
# imports
######################################
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
import uproot
# comstumized
from framework.GPUOperators import *
from framework.GPUStruct import *

def exclusiveCumsum(arr,dtype=np.uint32):
    neutral = 0
    res = np.cumsum(arr, dtype=dtype)

    return np.insert(res[:-1],0,neutral)

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


######################################
# Functions for AnalyzerInitCuda.py
######################################
def generate_struct_declaration(features, fConf, structName):
    '''
    This is used for events mask/in/internal/out SoA, 4 SoA in total.
    It is called 4 times in AnalyzerInitCuda.py in 'generate_cuda_struct_declaration'
    '''

    code = "// {} is auto-generated from csv file \n".format(structName)
    code += "struct " + structName + "{\n"

    # add features as arrays
    for f in features:

        if f in fConf.index:
            # use fConf.type if format
            fmt = fConf.loc[f,'type']
            # use fConf.isArray 
            isArray = fConf.loc[f,'isArray']==1

        elif ("cumsum" in f):
            # cumsum is always uint
            fmt = 'uint'
            # cumsum is always array
            isArray = True

        elif f=="nev":
            # nev is always int
            fmt = 'int'
            # nev is never array
            isArray = False

        else:
            raise RuntimeError("cannot generate_struct: type of '{}' is unknown".format(f))
      
        obj = "*"+f if isArray else f
        # convert bool to int because cuda doesn't support array of bool
        if fmt == "bool": fmt = "int"

        # add fmt obj to code
        code += "    {} {};\n".format(fmt, obj)
    
    # ending parenthesis 
    code += "};\n\n"
    return code





######################################
# Functions for AnalyzerInitData.py
######################################

# init_empty_host_soa
ctype2callable = {
    'bool' : np.bool, 
    'uint' : np.uint32, 
    'int'  : np.int32, 
    'float': np.float32
    }
def init_empty_host_soa(fConf, soa, nev):
    '''
    This is used to reset events internal/out SoA to zeros, 2 SoA in total.
    mask/in SoA is read directly from root file.
    It is called 2 times in AnalyzerInitData.py in 'init_events_internal' and 'init_events_out'
    '''
    # host SoA
    for f, conf in fConf.iterrows():

        fmt, rule, num = conf.type, conf.rule, conf.num
        # convert string to callable np.dtype
        fmt = ctype2callable[fmt]
        num = eval(num) if type(num) is str else num
        # rule is variable
        if rule == 'v':
            soa[f] = fmt(num)
        # rule is length
        if rule == 'l':
            soa[f] = np.zeros(nev * num, dtype=fmt)




# get_device_soa
dtype2callable = {
    np.dtype('bool')   : np.bool, 
    np.dtype('uint32') : np.uint32, 
    np.dtype('int32')  : np.int32, 
    np.dtype('float32'): np.float32
    }
def get_device_soa(features, soa):
    '''
    This is used for events mask/in/internal/out SoA, 4 SoA in total.
    It is called 4 times in AnalyzerInitData.py.
    '''
    objs = []

    for f in features:
        v = soa[f]
        if type(v)==np.ndarray:
            obj = "*"+f
            fmt = dtype2callable[v.dtype]
        else:
            # any non-array is int32
            obj = f
            fmt = np.int32
        objs.append((fmt,obj,v))

    return GPUStruct(objs)


