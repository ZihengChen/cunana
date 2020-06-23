
from framework.AnalyzerUtilities import *

from framework.Analyzer import Analyzer

class ExampleAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.profileEvents = True

        self.baseDir = '/home/zchen/Documents/Analysis/nanoblt'

        self.csvDir = '/ExampleAnalyzer/featureConfig'
        self.featureConfigCSVFiles = [
            self.baseDir + self.csvDir + '/inFeature.csv',
            self.baseDir + self.csvDir + '/internalFeature.csv',
            self.baseDir + self.csvDir + '/outFeature.csv'
            ]

        self.cuDir = '/ExampleAnalyzer/cuda'
        self.cuFiles = [
            self.baseDir + self.cuDir + '/mask.cu',
            self.baseDir + self.cuDir + '/objectSelection.cu',
            ]

        self.cuKernelsNames = [
            'knl_mask',
            'knl_objectSelection_electron', 
            'knl_objectSelection_muon'
            ]
        
        self.init_cuda()

        


    def object_selection(self):
        self.kernels.knl_objectSelection_electron(
            self.devents.get_ptr(), self.deventsInternal.get_ptr(), self.deventsOut.get_ptr(), 
            grid=(int(self.events.nev/1024)+1,1,1), block=(1024,1,1))

        self.kernels.knl_objectSelection_muon(
            self.devents.get_ptr(), self.deventsInternal.get_ptr(), self.deventsOut.get_ptr(), 
            grid=(int(self.events.nev/1024)+1,1,1), block=(1024,1,1))


    def event_selection(self):
        print("this is event selection")
        pass


    def postprocess(self):
        print("this is post process")
        pass



