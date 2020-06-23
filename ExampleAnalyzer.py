from Utilities import *


def config(self):
    self.featureConfigCSVFiles = [
        'ExampleAnalyzer_featureConfig/inFeature.csv',
        'ExampleAnalyzer_featureConfig/internalFeature.csv',
        'ExampleAnalyzer_featureConfig/outFeature.csv'
        ]


    self.cuFiles = [
        'ExampleAnalyzer_mask.cu',
        'ExampleAnalyzer_objectSelection.cu',
        ]

    self.cuKernelsNames = [
        'knl_mask',
        'knl_objectSelection_electron', 
        'knl_objectSelection_muon'
        ]



def object_selection(self):
    
    self.kernels.knl_objectSelection_electron(
        self.devents.get_ptr(), 
        self.deventsInternal.get_ptr(), 
        self.deventsOut.get_ptr(), 
        grid  = (int(self.events.nev/1024)+1,1,1), 
        block = (1024,1,1)
        )

    self.kernels.knl_objectSelection_muon(
        self.devents.get_ptr(), 
        self.deventsInternal.get_ptr(), 
        self.deventsOut.get_ptr(), 
        grid  = (int(self.events.nev/1024)+1,1,1), 
        block = (1024,1,1)
        )

def event_selection(self):
    pass


def postprocess(self):
    pass



