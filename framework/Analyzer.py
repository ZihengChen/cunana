
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
        self.profileEvents = False
        pass


    def process_infiles(self):
        pass


    def process_infile(self, infile, outfile):
        # load infile
        self.tree = uproot.open(infile)["Events"]
        self.get_mask() 
        self.init_events()

        # compute on gpu
        if self.profileEvents: start = time.time()
        self.copy_events_to_gpu()
        self.object_selection()
        self.event_selection()
        self.copy_events_from_gpu()
        self.clear_devents()
        if self.profileEvents: 
            print( "time totalGPU", time.time()-start)
        

        # stage outfile
        self.postprocess()
        self.store(outfile)
        self.clear_events()


    ################################################
    # CPU-GPU dataflow controlling methords
    ################################################

    def copy_events_to_gpu(self):
        self.devents.copy_to_gpu()
        self.deventsInternal.copy_to_gpu()
        self.deventsOut.copy_to_gpu()

    def copy_events_from_gpu(self):
        self.deventsOut.copy_from_gpu()

    def clear_devents(self):
        del self.devents
        del self.deventsInternal
        del self.deventsOut

    def store(self, outfile):
        print( "eventsOut is ", self.eventsOut)
        print("this is store data", outfile)
        pass

    def clear_events(self):
        self.tree = None
        self.events = None
        self.eventsInternal.clear()
        self.eventsOut.clear()
