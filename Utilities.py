
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
import uproot

from GPUOperators import *
from GPUStruct import *


global MAXNLEPTON
MAXNLEPTON = 4


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


dtype2callable = {
    np.dtype('bool')   : np.bool, 
    np.dtype('uint32') : np.uint32, 
    np.dtype('int32')  : np.int32, 
    np.dtype('float32'): np.float32
    }

ctype2callable = {
    'bool' : np.bool, 
    'uint' : np.uint32, 
    'int'  : np.int32, 
    'float': np.float32
    }



def generate_struct_declaration( features, fConf, structName):
    code = "// {} is auto-generated from csv file \n".format(structName)
    code += "struct " + structName + "{\n"

    # add features as arrays
    for f in features:

        if f in fConf.index:
            fmt = fConf.loc[f,'type']
            isArray = fConf.loc[f,'isArray']==1

        elif ("cumsum" in f):
            fmt = 'uint' 
            isArray = True

        elif f=="nev":
            fmt = 'int'
            isArray = False

        else:
            raise RuntimeError("cannot generate_struct: type of '{}' is unknown".format(f))
      
        obj = "*"+f if isArray else f
        # convert bool to int because cuda doesn't support array of bool
        if fmt == "bool": fmt = "int"


        code += "    {} {};\n".format(fmt, obj)
    
    code += "};\n\n"
    return code


def getDeviceSoA( features, SoA):
    objs = []

    for f in features:
        v = SoA[f]
        if type(v)==np.ndarray:
            obj = "*"+f
            fmt = dtype2callable[v.dtype]
        else:
            # any non-array is int32
            obj = f
            fmt = np.int32
        objs.append((fmt,obj,v))

    return GPUStruct(objs)