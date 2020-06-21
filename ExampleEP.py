
from EventProcessor import *
from ExampleEP import *



class ExampleEP(EventProcessor):

    from ExampleEP_objectSelection import \
        objectSelection_electrons, \
        objectSelection_muons

    from ExampleEP_eventSelection import \
        eventSelection_channelFilling_ee, \
        eventSelection_channelFilling_mumu, \
        eventSelection_leptonsP4Filling

    def __init__(self):
        super().__init__()
        self.channelsNames = ["ee", "mumu"]
        self.nchannels = len(self.channelsNames)

        self.knl_mask = cu.elementwise.ElementwiseKernel(
            arguments="int *mask, bool *HLT_Ele32_WPTight_Gsf, bool *HLT_IsoMu24, int *nElectron, int *nMuon",
            operation="mask[i] = (HLT_Ele32_WPTight_Gsf[i] || HLT_IsoMu24[i]) && (nElectron[i]>=2 || nMuon[i]>=2)"
            )

 
    def get_mask_for_events_in_file(self, infile):
        
        tree = uproot.open(infile)["Events"]
        featuresName = ['HLT_Ele32_WPTight_Gsf',"HLT_IsoMu24","nElectron","nMuon"]
        features = tree.arrays(featuresName, outputtype=DotDict, namedecode="utf-8")
        nin = len(features.nElectron)

        # # copy features to device
        # dfeatures = DotDict({k:cu.gpuarray.to_gpu(features[k]) for k in features})
        # del features # delete host features

        # # get mask
        # mask = pycuda.gpuarray.empty(nin, np.int32)
        # self.knl_mask(mask, dfeatures.HLT_Ele32_WPTight_Gsf, dfeatures.HLT_IsoMu24, dfeatures.nElectron, dfeatures.nMuon)
        # del dfeatures # delete device features

        # # get compaction
        # self.compaction = parallelCompact(mask).get()
        # mask = mask.get()
        # return mask


        # trigger mask
        mask1 = features.HLT_Ele32_WPTight_Gsf | features.HLT_IsoMu24 
        # minimum multiplicity mask
        mask2 = (features.nElectron>=2) | (features.nMuon>=2)
        # total
        mask = mask1 & mask2

        self.compaction = np.arange(nin)[mask]
        return mask


    def process_event(self): 
    
        self.objectSelection_electrons()
        self.objectSelection_muons()

        self.out.Channel = -1
        # 0) e-trigger, ee
        if self.event.HLT_Ele32_WPTight_Gsf and self.out.nElectrons>=2 and self.out.nMuons==0:
            self.eventSelection_channelFilling_ee()
            
        # 1) mu-trigger, mumu
        elif self.event.HLT_IsoMu24 and self.out.nElectrons==0 and self.out.nMuons>=2:
            self.eventSelection_channelFilling_mumu()