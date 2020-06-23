
from framework.AnalyzerUtilities import *


def get_mask(self):

    if self.profileEvents: start = time.time()
    # host SoA: read features
    evs = self.tree.arrays(self.inMaskFeatures, outputtype=DotDict, namedecode="utf-8")
    # host SoA: bool to int32 because cuda does not support array of bool
    evs.update( (k, evs[k].astype(np.int32)) for k in evs if evs[k].dtype == bool )
    # host SoA: add nev
    evs.nev = len(evs.luminosityBlock)

    # device SoA
    devs = get_device_soa(["nev"]+self.inMaskFeatures, soa=evs)
    devs.copy_to_gpu()

    # initiate mask as gpuarray
    mask = pycuda.gpuarray.empty(evs.nev, np.bool)
    # elementwise kernel for gpu array
    self.kernels.knl_mask(devs.get_ptr(), mask,  grid=(int(evs.nev/1024)+1,1,1), block=(1024,1,1))    
    
    # copy mask from gpu
    self.mask = mask.get()

    # print mask efficiency
    if self.profileEvents: 
        end = time.time()
        print( "time mask", end-start)

        nev2 = self.mask.sum()
        print("pass mask {}/{}, eff={:5.3f}".format(
            nev2, evs.nev, float(nev2)/float(evs.nev)) )

    # delete host and delete SoA
    del evs, devs





def init_events(self):
    
    # host SoA: read features
    if self.profileEvents: start = time.time()
    self.events = self.tree.arrays( 
        list(self.fConf.index),
        outputtype = DotDict, flatten=False, namedecode="utf-8" )
    if self.profileEvents: 
        print( "time init_events read tree", time.time()-start)   

    # host SoA: apply mask
    if self.profileEvents: start = time.time()
    self.events.update( 
        (k, self.events[k][self.mask]) 
        for k in self.events )
    del self.mask
    if self.profileEvents: 
        print( "time init_events apply mask", time.time()-start)


    # host SoA: flat jagged array
    if self.profileEvents: start = time.time()
    self.events.update( 
        (k, self.events[k].flatten()) 
        for k in self.events 
        if type(self.events[k]) is awkward.array.jagged.JaggedArray)
    if self.profileEvents: 
        print( "time init_events flat jaggedArray", time.time()-start)


    # host SoA: bool to int32 because cuda does not support array of bool
    if self.profileEvents: start = time.time()
    self.events.update( 
        (k, self.events[k].astype(np.int32)) 
        for k in self.events 
        if self.events[k].dtype == bool )
    if self.profileEvents: 
        print( "time init_events bool2int", time.time()-start)

    # host SoA: add cumsum
    if self.profileEvents: start = time.time()
    self.events.update( 
        ("cumsum_"+k, np.cumsum(self.events[k], dtype=np.uint32)) 
        for k in self.needCumsumFeatures )
    if self.profileEvents: 
        print( "time init_events cumsum", time.time()-start)
    
    # host SoA: add nev
    self.events.nev = len(self.events.luminosityBlock)
    

    # device SoA
    if self.profileEvents: start = time.time()
    self.devents = get_device_soa(
        ['nev']+self.inSelectionFeatures+self.cumsumedFeatures, soa=self.events )
    if self.profileEvents: 
        print( "time init_events device soa", time.time()-start)

    # also init internal variables and out variables 
    # Host SoA and device SoA
    if self.profileEvents: start = time.time()
    self.init_events_internal()
    self.init_events_out()
    if self.profileEvents: 
        print( "time init_events internal-out", time.time()-start)


def init_events_internal(self):
    # host SoA
    init_empty_host_soa(self.eventsInternalFConfig, soa=self.eventsInternal, nev=self.events.nev)
    # device SoA
    self.deventsInternal = get_device_soa(self.internalFeatures, soa=self.eventsInternal) 



def init_events_out(self):
    # host SoA
    init_empty_host_soa(self.eventsOutFConf, soa=self.eventsOut, nev=self.events.nev)
    # device SoA
    self.deventsOut = get_device_soa(self.outFeatures, soa=self.eventsOut) 
    


