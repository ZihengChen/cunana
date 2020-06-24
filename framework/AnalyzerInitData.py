
from framework.AnalyzerUtilities import *

def init_events(self):
    
    
    if self.verb: start = time.time()
    # host SoA: read features
    self.events = self.tree.arrays( 
        list(self.fConf.index),
        outputtype = DotDict, flatten=False, namedecode="utf-8" )
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events read_tree", 
            time.time()-start))



    
    if self.verb: start = time.time()
    # get mask
    mask = self.get_mask()
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events get_mask", 
            time.time()-start))




    if self.verb: start = time.time()
    # host SoA: apply mask
    self.events.update( 
        (k, self.events[k][mask]) 
        for k in self.events )
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events apply_mask", 
            time.time()-start))

    


    if self.verb: start = time.time()
    # host SoA: flat jagged array
    self.events.update( 
        (k, self.events[k].flatten()) 
        for k in self.events 
        if type(self.events[k]) is awkward.array.jagged.JaggedArray)
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events flat_JaggedArray", 
            time.time()-start))




    if self.verb: start = time.time()
    # host SoA: add cumsum
    self.events.update( 
        ("cumsum_"+k, exclusiveCumsum(self.events[k], dtype=np.uint32)) 
        for k in self.needCumsumFeatures )
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events cumsum", 
            time.time()-start))

    
    if self.verb: start = time.time()
    # host SoA: bool to int32 because cuda does not support array of bool
    self.events.update( 
        (k, self.events[k].astype(np.int32)) 
        for k in self.events 
        if self.events[k].dtype == bool )
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events bool2int", 
            time.time()-start))



    # host SoA: add nev
    self.events.nev = len(self.events.luminosityBlock)
    self.nev = len(self.events.luminosityBlock)



    if self.verb: start = time.time()
    # device SoA
    self.devents = get_device_soa(
        ['nev']+self.inSelectionFeatures+self.cumsumedFeatures, soa=self.events )
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events get_device_soa", 
            time.time()-start))



    if self.verb: start = time.time()
    # also init internal variables and out variables 
    # Host SoA and device SoA
    self.init_events_internal()
    self.init_events_out()
    # verb time
    if self.verb: 
        print( "{:35}: {:9.6f}s".format(
            "time init_events internal_out", 
            time.time()-start))



def get_mask(self):

    self.events.nev = len(self.events.luminosityBlock)
    # host SoA: bool to int32 because cuda does not support array of bool
    self.events.update( 
        (k, self.events[k].astype(np.int32)) 
        for k in self.inMaskFeatures 
        if self.events[k].dtype == bool )

    # device SoA
    devs = get_device_soa(["nev"]+self.inMaskFeatures, soa=self.events)
    devs.copy_to_gpu()

    # initiate mask as gpuarray
    dmask = pycuda.gpuarray.empty(self.events.nev, np.bool)
    # elementwise kernel for gpu array
    self.kernels.knl_mask(
        devs.get_ptr(), dmask,  
        grid=(int(self.events.nev/1024)+1,1,1), 
        block=(1024,1,1))    
    
    # copy mask from gpu
    mask = dmask.get()

    self.nevRaw = self.events.nev

    del devs, self.events['nev']
    return mask


def init_events_internal(self):
    # host SoA
    init_empty_host_soa(
        self.eventsInternalFConfig, 
        soa=self.eventsInternal, 
        nev=self.events.nev)
    # device SoA
    self.deventsInternal = get_device_soa(
        self.internalFeatures, 
        soa=self.eventsInternal) 



def init_events_out(self):
    # host SoA
    init_empty_host_soa(
        self.eventsOutFConf, 
        soa=self.eventsOut, 
        nev=self.events.nev)
    # device SoA
    self.deventsOut = get_device_soa(
        self.outFeatures, 
        soa=self.eventsOut) 
    

# def get_mask(self):
    # # host SoA: read features
    # evs = self.tree.arrays(self.inMaskFeatures, outputtype=DotDict, namedecode="utf-8")
    # # host SoA: bool to int32 because cuda does not support array of bool
    # evs.update( (k, evs[k].astype(np.int32)) for k in evs if evs[k].dtype == bool )
    # # host SoA: add nev
    # evs.nev = len(evs.luminosityBlock)

    # # device SoA
    # devs = get_device_soa(["nev"]+self.inMaskFeatures, soa=evs)
    # devs.copy_to_gpu()

    # # initiate mask as gpuarray
    # mask = pycuda.gpuarray.empty(evs.nev, np.bool)
    # # elementwise kernel for gpu array
    # self.kernels.knl_mask(devs.get_ptr(), mask,  grid=(int(evs.nev/1024)+1,1,1), block=(1024,1,1))    
    
    # # copy mask from gpu
    # self.mask = mask.get()

    # # print mask efficiency

    #     nev2 = self.mask.sum()
    #     print("pass mask {}/{}, eff={:5.3f}".format(
    #         nev2, evs.nev, float(nev2)/float(evs.nev)) )

    # # delete host and delete SoA
    # del evs, devs

