
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-06-16 17:01:45
# @modify date 2020-06-16 17:01:45
from Analyzer import *
from Utility import *

class MPMaster():
    def __init__(self, nthreads, indir, outdir):
        
        self.nthreads= nthreads
        self.indir   = indir
        self.outdir  = outdir

        if not os.path.exists(self.outdir):
            os.system('mkdir -p '+self.outdir)
        
    
        nanoaods = glob.glob(indir + "/*.root")
        nInfilePerJob = (len(nanoaods)//nthreads)
        self.list_of_infiles = [nanoaods[i:i+nInfilePerJob] for i in range(nthreads)]
        
    def run_analyzer(self, id):
        ana     = Analyzer()
        infiles = self.list_of_infiles[id]
        outfile = self.outdir+'/output_{}.h5'.format(id)
        ana.process_infiles(infiles, outfile)

    

    def multiprocess(self):
        pool = multiprocessing.Pool(self.nthreads)
        pool.map(self.run_analyzer, range(self.nthreads))
        # for i in range(self.nthreads):
        #     pool.apply_async(self.run_analyzer, args=(i,)) 

        # jobs = []
        # for i in range(self.nthreads):
        #     p = multiprocessing.Process(target=self.run_analyzer, args=(i,))
        #     jobs.append(p)
        
        # for p in jobs:
        #     p.start()
        
        # for p in jobs:
        #     p.join()
        