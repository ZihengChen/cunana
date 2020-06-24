
from framework.AnalyzerUtilities import *


class Analyzer():
    
    from framework.AnalyzerInitCuda import \
        init_cuda,\
        load_feature_config,\
        generate_cuda_struct_declaration,\
        compile_cuda_kernels
    
    from framework.AnalyzerInitData import \
        get_mask,\
        init_events,\
        init_events_internal,\
        init_events_out

        
    def __init__(self):
        self.verb = False
        pass


    def process_infiles(self, infiles, outfile):
        if(os.path.exists(outfile)):
            os.remove(outfile)

        for infile in infiles:
            start = time.time()
            self.process_infile(infile, outfile)
            print( "--- nRaw = {:6.1f}k, n = {:6.1f}k, totalTime = {:9.6f}s --- \n".format(
                self.nevRaw/1000, self.nev/1000, time.time()-start) )


    def process_infile(self, infile, outfile):
        # load infile
        self.tree = uproot.open(infile)["Events"]
        self.init_events()

        # compute on gpu
        if self.verb: start = time.time()
        self.copy_events_to_gpu()
        self.object_selection()
        self.event_selection()
        self.copy_events_from_gpu()
        self.clear_devents()
        if self.verb: 
            print( "{:35}: {:9.6f}s".format(
                "time total gpu", 
                time.time()-start))
        

        # stage outfile
        if self.verb: start = time.time()
        self.postprocess()
        if self.verb: 
            print( "{:35}: {:9.6f}s".format(
                "time postprocess", 
                time.time()-start))


        if self.verb: start = time.time()
        self.store(outfile)
        if self.verb: 
            print( "{:35}: {:9.6f}s".format(
                "time store to hdf", 
                time.time()-start))

        # self.clear_events()


    ################################################
    # CPU-GPU dataflow controlling methords
    ################################################

    def copy_events_to_gpu(self):
        self.devents.copy_to_gpu()
        self.deventsInternal.copy_to_gpu()
        self.deventsOut.copy_to_gpu()

    def copy_events_from_gpu(self):
        self.deventsInternal.copy_from_gpu()
        self.deventsOut.copy_from_gpu()

    def clear_devents(self):
        del self.devents
        del self.deventsInternal
        del self.deventsOut

    def store(self, outfile):
        
        for i,df in enumerate(self.channelDataframes):
            if len(df)>0:
                df.to_hdf(outfile, key='ch'+str(i), 
                append=True,  data_columns=True,format='table')
        

    def clear_events(self):
        self.tree = None
        self.events = None
        self.eventsInternal.clear()
        self.eventsOut.clear()
