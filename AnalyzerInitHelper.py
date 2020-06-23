
from Utilities import *

def load_feature_config(self):

    # mask features and in features: config

    # df
    self.fConf = pd.read_csv(self.featureConfigCSVFiles[0], index_col="featureName")
    # lists
    self.needCumsumFeatures   = list(self.fConf.query("needCumsum==1").index)
    self.cumsumedFeatures     = ["cumsum_"+k for k in self.needCumsumFeatures]
    self.inMaskFeatures       = list(self.fConf.query("inMask==1").index)
    self.inSelectionFeatures  = list(self.fConf.query("inSelection==1").index)

    # mask features: generate struct declaration
    self.codeMaskEventsStruct = generate_struct_declaration(
        ['nev']+self.inMaskFeatures, self.fConf, structName="MaskEvents")

    # in features: generate struct declaration
    self.codeEventsStruct = generate_struct_declaration(
        ['nev']+self.inSelectionFeatures+self.cumsumedFeatures, self.fConf, structName="Events")




    # internal features: config
    self.eventsInternal = DotDict()
    # df
    self.eventsInternalFConfig = pd.read_csv(self.featureConfigCSVFiles[1], index_col='featureName')
    # list
    self.internalFeatures = list(self.eventsInternalFConfig.index)
    # internal features: generate struct declaration
    self.codeEventsInternalStruct = generate_struct_declaration(
        self.internalFeatures, self.eventsInternalFConfig, structName="EventsInternal")




    # out features: config
    self.eventsOut = DotDict()
    # df
    self.eventsOutFConf = pd.read_csv(self.featureConfigCSVFiles[2], index_col='featureName')
    # list
    self.outFeatures = list(self.eventsOutFConf.index)
    # out features: generate struct declaration
    self.codeEventsOutStruct = generate_struct_declaration(
        self.outFeatures, self.eventsOutFConf, structName="EventsOut")





def compile_cuda_kernels(self):
    code = ""
    # add struct declarations
    code += self.codeMaskEventsStruct 
    code += self.codeEventsStruct
    code += self.codeEventsInternalStruct
    code += self.codeEventsOutStruct

    # read code from src files
    for cufile in self.cuFiles:
        f = open(cufile,"r") 
        code += f.read()
        f.close()

    # save total cu file
    f = open("temp.cu", "w")
    f.write(code)
    f.close()

    # complie cuda code
    module = cu.compiler.SourceModule( code )
    self.kernels = DotDict({ name: module.get_function(name) for name in self.cuKernelsNames })





def get_mask(self):

    # host SoA: read features
    evs = self.tree.arrays(self.inMaskFeatures, outputtype=DotDict, namedecode="utf-8")
    # host SoA: bool to int32 because cuda does not support array of bool
    evs.update( (k, evs[k].astype(np.int32)) for k in evs if evs[k].dtype == bool )
    # host SoA: add nev
    evs.nev = len(evs.luminosityBlock)

    # device SoA
    devs = getDeviceSoA(["nev"]+self.inMaskFeatures, SoA=evs)
    devs.copy_to_gpu()

    # initiate mask as gpuarray
    mask = pycuda.gpuarray.empty(evs.nev, np.bool)
    # elementwise kernel for gpu array
    self.kernels.knl_mask(devs.get_ptr(), mask,  grid=(int(evs.nev/1024)+1,1,1), block=(1024,1,1))
    # delete host and delete SoA
    del evs, devs
    # copy mask from gpu
    self.mask = mask.get()





def init_events(self):

    # host SoA: read features
    self.events = self.tree.arrays( 
        list(self.fConf.index),
        outputtype=DotDict, flatten=False, namedecode="utf-8" )

    # host SoA: apply mask
    self.events.update( 
        (k, self.events[k][self.mask]) 
        for k in self.events )

    # host SoA: flat jagged array
    self.events.update( 
        (k, self.events[k].flatten()) 
        for k in self.events 
        if type(self.events[k]) is awkward.array.jagged.JaggedArray)

    # host SoA: bool to int32 because cuda does not support array of bool
    self.events.update( 
        (k, self.events[k].astype(np.int32)) 
        for k in self.events 
        if self.events[k].dtype == bool )

    # host SoA: add cumsum
    self.events.update( 
        ("cumsum_"+k, np.cumsum(self.events[k], dtype=np.uint32)) 
        for k in self.needCumsumFeatures )

    # host SoA: add nev
    self.events.nev = len(self.events.luminosityBlock)
    

    # device SoA
    self.devents = getDeviceSoA(
        ['nev']+self.inSelectionFeatures+self.cumsumedFeatures, SoA=self.events )


    # also init internal variables and out variables 
    self.init_events_internal()
    self.init_events_out()






def init_events_internal(self):
    # host SoA
    for key, conf in self.eventsInternalFConfig.iterrows():

        fmt, rule, num = conf.type, conf.rule, conf.num
        # convert string to callable np.dtype
        fmt = ctype2callable[fmt]
        num = eval(num) if type(num) is str else num
        # rule is variable
        if rule == 'v':
            self.eventsInternal[key] = fmt(num)
        # rule is length
        if rule == 'l':
            self.eventsInternal[key] = np.zeros(self.events.nev * num, dtype=fmt)
        
    # device SoA
    self.deventsInternal = getDeviceSoA(self.internalFeatures, SoA=self.eventsInternal) 




def init_events_out(self):
    # host SoA
    for key, conf in self.eventsOutFConf.iterrows():

        fmt, rule, num = conf.type, conf.rule, conf.num
        # convert string to callable np.dtype
        fmt = ctype2callable[fmt]
        num = eval(num) if type(num) is str else num
        # rule is variable
        if rule == 'v':
            self.eventsOut[key] = fmt(num)
        # rule is length
        if rule == 'l':
            self.eventsOut[key] = np.zeros(self.events.nev * num, dtype=fmt)

    # device SoA
    self.deventsOut = getDeviceSoA(self.outFeatures, SoA=self.eventsOut) 