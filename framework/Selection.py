
from framework.Utilities import *


def object_selection(self):
    self.cuKernels['knl_objectSelection_electron'](
        self.soaManager.deventsIn.get_ptr(), 
        self.soaManager.deventsMid.get_ptr(), 
        self.soaManager.deventsOut.get_ptr(), 
        grid = (int(self.soaManager.nev/1024)+1,1,1), 
        block = (1024,1,1))

    self.cuKernels['knl_objectSelection_muon'](
        self.soaManager.deventsIn.get_ptr(), 
        self.soaManager.deventsMid.get_ptr(), 
        self.soaManager.deventsOut.get_ptr(), 
        grid = (int(self.soaManager.nev/1024)+1,1,1), 
        block=(1024,1,1))




def event_selection(self):
    self.nChannel = 2
    self.cuKernels['knl_eventSelection'](
        self.soaManager.deventsIn.get_ptr(), 
        self.soaManager.deventsMid.get_ptr(), 
        self.soaManager.deventsOut.get_ptr(), 
        grid = (int(self.soaManager.nev/1024)+1,1,1), 
        block = (1024,1,1))

