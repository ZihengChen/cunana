
from framework.Utilities import *

def eventsin_load(self, infile):
    

    if self.verb: start = time.time()
    tree = uproot.open(infile)["Events"]
    # load soa
    self.eventsIn = tree.arrays( 
        self.fconfHandler.fconfInLs, 
        outputtype = DotDict, 
        flatten = False, 
        namedecode = "utf-8" )

    # verb time
    if self.verb: print( 
        "{:35}: {:9.6f}s".format(
            "time eventsin_load", 
            time.time()-start))



def eventsin_get_mask(self, knl_mask):
    if self.verb: start = time.time()

    # temporarily add nev
    self.eventsIn.nev = len(self.eventsIn.luminosityBlock)
    self.nevRaw = len(self.eventsIn.luminosityBlock)

    # bool to int32 because cuda does not support array of bool        
    self.eventsIn.update( 
        (k, self.eventsIn[k].astype(np.int32)) 
        for k in self.fconfHandler.fconfInLsMaskEventsIn 
        if k!='nev' and self.eventsIn[k].dtype == bool )

    # device SoA
    self.get_dmaskeventsin()
    self.dmaskeventsIn.copy_to_gpu()

    # initiate mask as gpuarray
    dmask = pycuda.gpuarray.empty(self.eventsIn.nev, np.bool)

    # elementwise kernel for gpu array
    knl_mask(
        self.dmaskeventsIn.get_ptr(), dmask,  
        grid = (int(self.eventsIn.nev/1024)+1,1,1), 
        block = (1024,1,1))    

    # copy mask from gpu
    self.mask = dmask.get()

    # remove nev from eventsIn because it is nevRaw
    del self.eventsIn.nev

    # verb time
    if self.verb: print( 
        "{:35}: {:9.6f}s".format(
            "time eventsin_get_mask", 
            time.time()-start))



def eventsin_apply_mask(self):
    if self.verb: start = time.time()

    # apply mask
    self.eventsIn.update( 
        (k, v[self.mask]) 
        for k,v in self.eventsIn.items())

    # verb time
    if self.verb: print(
        "{:35}: {:9.6f}s".format(
            "time eventsin_apply_mask", 
            time.time()-start))




def eventsin_flat_jaggedarray(self):
    if self.verb: start = time.time()

    # flat jagged array
    self.eventsIn.update(
        (k, v.flatten()) 
        for k,v in self.eventsIn.items() 
        if type(v) is awkward.array.jagged.JaggedArray )

    # verb time
    if self.verb: print( 
        "{:35}: {:9.6f}s".format(
            "time eventsin_flat_jaggedarray", 
            time.time()-start))




def eventsin_add_cumsum(self):
    if self.verb: start = time.time()
    # add cumsum
    self.eventsIn.update(
        ("cumsum_"+k, exclusiveCumsum(self.eventsIn[k], dtype=np.uint32)) 
        for k in self.fconfHandler.fconfInLsNeedCumsum )
        # exclusiveCumsum is customized from numpy.cumsum

    # verb time
    if self.verb: print( 
        "{:35}: {:9.6f}s".format(
            "time eventsin_add_cumsum", 
            time.time()-start))




def eventsin_convert_bool2int(self):
    if self.verb: start = time.time()
    # bool to int32 because cuda does not support array of bool
    self.eventsIn.update(
        (k, v.astype(np.int32)) 
        for k,v in self.eventsIn.items() 
        if v.dtype == bool )

    # verb time
    if self.verb: print(
        "{:35}: {:9.6f}s".format(
            "time eventsin_convert_bool2int", 
            time.time()-start))




def eventsin_add_nev(self):
    # host SoA: add nev
    self.eventsIn.nev = len(self.eventsIn.luminosityBlock)
    self.nev = len(self.eventsIn.luminosityBlock)
