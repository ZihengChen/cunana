from framework.Initiator import Initiator
from framework.SoaManager import SoaManager
from framework.Utilities import *


class Processor():
    from framework.Selection import \
        object_selection,\
        event_selection


    def __init__(self, initiator):
        self.initiator = initiator
        self.soaManager = SoaManager()
        self.soaManager.set_fconfHandler(self.initiator.fconfHandler)
        self.cuKernels = self.initiator.cuHandler.cuKernels

        self.verb = True
        self.soaManager.verb = self.verb
        
    
    def process_infiles(self, infiles, outfile):
        if(os.path.exists(outfile)):
            os.remove(outfile)

        for infile in infiles:
            start = time.time()
            self.process_infile(infile, outfile)
            print( "--- nRaw = {:6.1f}k, n = {:6.1f}k, totalTime = {:9.6f}s --- \n".format(
                self.soaManager.nevRaw/1000, self.soaManager.nev/1000, time.time()-start) )


    def process_infile(self, infile, outfile):
        # load infile
        self.load_events(infile)

        # compute on gpu
        if self.verb: start = time.time()
        self.copy_events_to_gpu()
        self.object_selection()
        self.event_selection()
        self.copy_events_from_gpu()
        if self.verb: print( 
            "{:35}: {:9.6f}s".format(
                "time total gpu", 
                time.time()-start))
        

        # stage outfile
        # postprocess
        if self.verb: start = time.time()
        self.postprocess()
        if self.verb: print( 
            "{:35}: {:9.6f}s".format(
                "time postprocess", 
                time.time()-start))

        # store
        if self.verb: start = time.time()
        self.store(outfile)
        if self.verb: 
            print( "{:35}: {:9.6f}s".format(
                "time store to hdf", 
                time.time()-start))
        
        # clear
        self.soaManager.clear()
        
    def load_events(self, infile):
        # host SoA
        self.soaManager.get_eventsin(infile, self.cuKernels['knl_mask'])
        self.soaManager.get_eventsmid()
        self.soaManager.get_eventsout()

        # device SoA
        self.soaManager.get_deventsin()
        self.soaManager.get_deventsmid()
        self.soaManager.get_deventsout()


    def copy_events_to_gpu(self):
        self.soaManager.deventsIn.copy_to_gpu()
        self.soaManager.deventsMid.copy_to_gpu()
        self.soaManager.deventsOut.copy_to_gpu()

    def copy_events_from_gpu(self):
        self.soaManager.deventsMid.copy_from_gpu()
        self.soaManager.deventsOut.copy_from_gpu()



    def postprocess(self):
        rawIndex = np.arange(self.soaManager.nev)

        self.channelDataframes = []

        for i in range(self.nChannel):
            rawIndexChi = rawIndex[self.soaManager.eventsOut.channel==i]
            chi = {'rawIndex': rawIndexChi}
            # events features that need save
            for k in self.initiator.fconfHandler.fconfInLsNeedSave:
                chi[k] = self.soaManager.eventsIn[k][rawIndexChi]

            # all eventsOut features
            for k in self.soaManager.eventsOut:
                chi[k] = self.soaManager.eventsOut[k][rawIndexChi]


            self.channelDataframes.append( pd.DataFrame(chi) )


    def store(self, outfile):  
        for i,df in enumerate(self.channelDataframes):
            if len(df)>0:
                df.to_hdf(outfile, key='ch'+str(i), 
                append=True,  data_columns=True,format='table')
        
