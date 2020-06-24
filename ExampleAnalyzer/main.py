
from framework.AnalyzerUtilities import *

from framework.Analyzer import Analyzer

class ExampleAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.verb = False

        self.baseDir = '/home/zchen/Documents/Analysis/nanoblt'

        self.csvDir = '/ExampleAnalyzer/featureConfig'
        self.featureConfigCSVFiles = [
            self.baseDir + self.csvDir + '/inFeature.csv',
            self.baseDir + self.csvDir + '/internalFeature.csv',
            self.baseDir + self.csvDir + '/outFeature.csv'
            ]

        self.cuDir = '/ExampleAnalyzer/cuda'
        self.cuFiles = [
            self.baseDir + self.cuDir + '/lorentz.cu',
            self.baseDir + self.cuDir + '/mask.cu',
            self.baseDir + self.cuDir + '/objectSelection.cu',
            self.baseDir + self.cuDir + '/eventSelection.cu',
            ]

        self.cuKernelsNames = [
            'knl_mask',
            'knl_objectSelection_electron', 
            'knl_objectSelection_muon',
            'knl_eventSelection'
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
        self.nChannel = 2
        self.kernels.knl_eventSelection(
            self.devents.get_ptr(), self.deventsInternal.get_ptr(), self.deventsOut.get_ptr(), 
            grid=(int(self.events.nev/1024)+1,1,1), block=(1024,1,1))


    def postprocess(self):
        rawIndex = np.arange(self.events.nev)

        self.channelDataframes = []

        for i in range(self.nChannel):
            rawIndexChi = rawIndex[self.eventsOut.channel==i]
            chi = {'rawIndex': rawIndexChi}
            # events features that need save
            for k in self.needSaveFeatures:
                chi[k] = self.events[k][rawIndexChi]

            # all eventsOut features
            for k in self.eventsOut:
                chi[k] = self.eventsOut[k][rawIndexChi]


            self.channelDataframes.append( pd.DataFrame(chi) )



