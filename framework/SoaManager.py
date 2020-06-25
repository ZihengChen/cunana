
from framework.Utilities import *

class SoaManager():

    from framework.SoaManagerTools import \
        eventsin_load,\
        eventsin_get_mask,\
        eventsin_apply_mask,\
        eventsin_flat_jaggedarray,\
        eventsin_add_cumsum,\
        eventsin_convert_bool2int,\
        eventsin_add_nev


    def __init__(self):

    
        self.mask = None
        self.eventsIn = DotDict()
        self.eventsMid = DotDict()
        self.eventsOut = DotDict()

        self.verb = True
        self.typeCallable = {
            'bool' : np.bool, 
            'uint' : np.uint32, 
            'int'  : np.int32, 
            'float': np.float32,

            np.dtype('bool')   : np.bool, 
            np.dtype('uint32') : np.uint32, 
            np.dtype('int32')  : np.int32, 
            np.dtype('float32'): np.float32,
            }


    def set_fconfHandler(self, fconfHandler):
        self.fconfHandler = fconfHandler
        

    def clear(self):
        self.mask = None
        self.eventsIn.clear()
        self.eventsMid.clear()
        self.eventsOut.clear()
        try:
            del self.dmaskeventsIn
            del self.deventsIn
            del self.deventsMid
            del self.deventsOut
        except:
            pass
        
    
    # host SoA
    def get_eventsin(self, infile, knl_mask):
        self.eventsin_load(infile)
        self.eventsin_get_mask(knl_mask)
        self.eventsin_apply_mask()
        self.eventsin_flat_jaggedarray()
        self.eventsin_add_cumsum()
        self.eventsin_convert_bool2int()
        self.eventsin_add_nev()


    def get_eventsmid(self):
        self.prepare_empty_host_soa(
            hostSoa = self.eventsMid, 
            fConfDf = self.fconfHandler.fconfMidDf, 
            nev = self.eventsIn.nev)

    def get_eventsout(self):
        self.prepare_empty_host_soa(
            hostSoa = self.eventsOut, 
            fConfDf = self.fconfHandler.fconfOutDf, 
            nev = self.eventsIn.nev)



    # device SoA
    def get_dmaskeventsin(self):
        self.dmaskeventsIn = self.get_device_soa(
            hostSoa = self.eventsIn,
            fconfLs = self.fconfHandler.fconfInLsMaskEventsIn)


    def get_deventsin(self):
        self.deventsIn = self.get_device_soa(
            hostSoa = self.eventsIn,
            fconfLs = self.fconfHandler.fconfInLsEventsIn)


    def get_deventsmid(self):
        self.deventsMid = self.get_device_soa(
            hostSoa = self.eventsMid,
            fconfLs = self.fconfHandler.fconfMidLs)
    
    def get_deventsout(self):
        self.deventsOut = self.get_device_soa(
            hostSoa = self.eventsOut,
            fconfLs = self.fconfHandler.fconfOutLs)



    ################################
    # helper functions for host SoA
    ################################
    def prepare_empty_host_soa(self, hostSoa, fConfDf, nev):
        '''
        This is used to reset events internal/out SoA to zeros, 2 SoA in total.
        mask/in SoA is read directly from root file.
        It is called 2 times in AnalyzerInitData.py in 'init_events_internal' and 'init_events_out'
        '''
        # host SoA
        for f, conf in fConfDf.iterrows():

            fmt, rule, num = conf.type, conf.rule, conf.num
            # convert string to callable np.dtype
            fmt = self.typeCallable[fmt]
            num = eval(num) if type(num) is str else num
            # rule is variable
            if rule == 'v':
                hostSoa[f] = fmt(num)
            # rule is length
            if rule == 'l':
                hostSoa[f] = np.zeros(nev * num, dtype=fmt)



    ################################
    # helper functions for device SoA
    ################################
    def get_device_soa(self, hostSoa, fconfLs):
        '''
        This is used for events mask/in/internal/out SoA, 4 SoA in total.
        It is called 4 times in AnalyzerInitData.py.
        '''
        objs = []

        for f in fconfLs:
            v = hostSoa[f]
            if type(v)==np.ndarray:
                obj = "*"+f
                fmt = self.typeCallable[v.dtype]
            else:
                # any non-array is int32
                obj = f
                fmt = np.int32
            objs.append((fmt,obj,v))

        return GPUStruct(objs)

