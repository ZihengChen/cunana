
from Utilities import *


class Analyzer():
    
    from AnalyzerInitHelper import \
        load_feature_config,\
        compile_cuda_kernels,\
        get_mask,\
        init_events,\
        init_events_internal,\
        init_events_out


    from ExampleAnalyzer import \
        config,\
        object_selection,\
        event_selection,\
        postprocess

        
    def __init__(self):
        self.config()
        self.load_feature_config()
        self.compile_cuda_kernels()
        

    def process_infiles(self):
        pass

    def process_infile(self, infile, outfile):
        # load infile
        self.tree = uproot.open(infile)["Events"]
        self.get_mask() 
        self.init_events()

        # compute on gpu
        self.copy_events_to_gpu()
        self.object_selection()
        self.event_selection()
        self.copy_events_from_gpu()
        self.clear_devents()
        self.postprocess()

        # stage outfile
        self.store(outfile)
        self.clear_events()


    ########################
    # dataflow managers
    ########################
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
        pass

    def clear_events(self):
        self.tree = None
        self.events = None
        self.eventsInternal.clear()
        self.eventsOut.clear()
