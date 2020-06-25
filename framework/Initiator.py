from framework.CuHandler import CuHandler
from framework.FconfHandler import FconfHandler


class Initiator():
    def __init__(self):
        self.baseDir = '/home/zchen/Documents/Analysis/cunana/'
        self.set_fconfHandler()
        self.set_cuHandler()
        
    
    def set_fconfHandler(self):    
        self.fconfHandler = FconfHandler(self.baseDir+'fconf/')
        

    def set_cuHandler(self):
        self.cuHandler = CuHandler(self.baseDir+'cuda/')

        self.cuHandler.generate_cu_struct( "MaskEventsIn",
            fconfLs = self.fconfHandler.fconfInLsMaskEventsIn, 
            fconfDf = self.fconfHandler.fconfInDf)

        self.cuHandler.generate_cu_struct( "EventsIn",
            fconfLs = self.fconfHandler.fconfInLsEventsIn,
            fconfDf = self.fconfHandler.fconfInDf)

        self.cuHandler.generate_cu_struct( "EventsMid",
            fconfLs = self.fconfHandler.fconfMidLs,
            fconfDf = self.fconfHandler.fconfMidDf)

        self.cuHandler.generate_cu_struct( "EventsOut",
            fconfLs = self.fconfHandler.fconfOutLs, 
            fconfDf = self.fconfHandler.fconfOutDf)

        self.cuHandler.compile_cu_module(
            cuFiles = [
                'lorentz.cu',
                'mask.cu', 
                'objectSelection.cu', 
                'eventSelection.cu',
            ])

        self.cuHandler.get_cu_kernels(
            cuKernelsNames = [
                'knl_mask',
                'knl_objectSelection_electron', 
                'knl_objectSelection_muon',
                'knl_eventSelection',
            ])

