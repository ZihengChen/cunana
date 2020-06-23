import numpy as np

# cuda
import pycuda as cu
import pycuda.autoinit
import pycuda.gpuarray
import pycuda.scan
import pycuda.elementwise
import pycuda.compiler
import pycuda.driver as cuda


knl_exsumscan = cu.scan.ExclusiveScanKernel(np.int32, "a+b", neutral=0)

knl_compact = cu.compiler.SourceModule("""
    __global__ void compact(int *rawindex, int *exsumscan, int *mask, const int n) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i<n) if(mask[i]==1) rawindex[exsumscan[i]] = i;
        }
    """).get_function("compact")

def parallelCompact(mask):
    nin = len(mask)
    # get exclusive sum scan
    exsumscan = knl_exsumscan(mask.copy())
    # get rawindex
    nout = int(cu.gpuarray.sum(mask).get()) 
    rawindex = cu.gpuarray.empty(nout, np.int32)
    n = cu.gpuarray.to_gpu(np.array(nin,dtype=np.int32))
    knl_compact(rawindex, exsumscan, mask, n, grid  = (int(nin/1024)+1,1,1), block=(1024,1,1))
    return rawindex