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