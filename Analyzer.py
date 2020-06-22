from gpustruct import GPUStruct
from Utility import *

class Analyzer():
    def __init__(self):
        self.fConf = pd.concat(pd.read_csv(f,index_col="featureName") for f in glob.glob("featureConfig/*.csv"))
    
        self.needCumsum_features   = list(self.fConf.query("needCumsum==1").index)
        self.inMask_features       = list(self.fConf.query("inMask==1").index)
        self.inSelection_features  = list(self.fConf.query("inSelection==1").index) \
                                   + ["cumsum_"+f for f in self.needCumsum_features]
        self.generate_declaration_of_event_structure_cc()

    def generate_declaration_of_event_structure_cc(self):

        code = "struct Events {\n"
        
        for f in self.inSelection_features:
            if "cumsum" in f:
                fmt = 'int' 
            else:
                fmt = self.fConf.loc[f,'type']
            # cuda does not support array of bool
            # so we have to convert bool to int
            if fmt == "bool":
                fmt = "int"

            code += "    {} *{};\n".format(fmt, f)

        code += "    int nev;\n"
        code += "};\n"
        self.code_event_struct = code


    def process_infiles(self):
        pass


    def set_infile(self, infile):
        self.infile = infile
        self.tree = uproot.open(infile)["Events"]
        self.get_mask()
        # read features
        self.events = self.tree.arrays( list(self.fConf.index),outputtype=DotDict, flatten=False, namedecode="utf-8" )
        # apply mask
        self.events.update( (k, self.events[k][self.mask]) for k in self.events)
        # flat jagged array
        self.events.update( (k, self.events[k].flatten()) for k in self.events if type(self.events[k]) is awkward.array.jagged.JaggedArray)
        # bool to int32 because cuda does not support array of bool
        self.events.update( (k, self.events[k].astype(np.int32)) for k in self.events if self.events[k].dtype == bool )
        # add cumsum
        self.events.update( ("cumsum_"+f, np.cumsum(self.events[f], dtype=np.uint32)) for f in self.needCumsum_features )
        # add nev
        self.events.nev = len(self.events.event)
        

        # create devents
        deventlist = [( dtype2callable[self.events[f].dtype], "*"+f, self.events[f]) for f in self.inSelection_features]
        deventlist.append((np.int32, "nev", self.events.nev))
        self.devents = GPUStruct( deventlist )
        # copy to gpu
        self.devents.copy_to_gpu()

    def clear_infile(self):
        self.events = None
        del self.devents
    

    def process_infile(self):
        self.selection()
        self.postprocess()
        self.store()



    def get_mask(self):

        evts = self.tree.arrays(self.inMask_features, outputtype=DotDict, namedecode="utf-8")
        mask1 = (evts.HLT_Ele32_WPTight_Gsf) | (evts.HLT_IsoMu24)
        mask2 = (evts.nElectron>=2) | (evts.nMuon>=2)
        self.mask = mask1 & mask2

        # # copy features to device
        # nin = len(features.nElectron)
        # dfeatures = DotDict({k:cu.gpuarray.to_gpu(features[k]) for k in features})
        # del features # delete host features

        # # get mask
        # mask = pycuda.gpuarray.empty(nin, np.int32)
        # self.knl_mask(mask, dfeatures.HLT_Ele32_WPTight_Gsf, dfeatures.HLT_IsoMu24, dfeatures.nElectron, dfeatures.nMuon)
        # del dfeatures # delete device features

        # # compact raw index
        # self.rawindex = parallelCompact(mask).get()
        # mask = mask.get()
        # return mask



  
    def object_selection(self):
        knl_objectSelection_electrons(
            self.devents.get_ptr(), 
            grid  = (int(self.events.nev/1024)+1,1,1), 
            block = (1024,1,1)
            )
        

    def postprocess(self):
        pass

    def store(self):
        pass
