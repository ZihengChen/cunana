


from framework.AnalyzerUtilities import *


def init_cuda(self):
    self.load_feature_config()
    self.generate_cuda_struct_declaration()
    self.compile_cuda_kernels()


        

def load_feature_config(self):
    print("--- load_feature_config from .csv files ---")

    # mask features, in features
    # df
    self.fConf = pd.read_csv(self.featureConfigCSVFiles[0], index_col="featureName")
    print("load inFeatures csv: ", self.featureConfigCSVFiles[0])
    # lists
    self.needCumsumFeatures   = list(self.fConf.query("needCumsum==1").index)
    self.cumsumedFeatures     = ["cumsum_"+k for k in self.needCumsumFeatures]
    self.inMaskFeatures       = list(self.fConf.query("inMask==1").index)
    self.inSelectionFeatures  = list(self.fConf.query("inSelection==1").index)


    # internal features
    self.eventsInternal = DotDict()
    # df
    self.eventsInternalFConfig = pd.read_csv(self.featureConfigCSVFiles[1], index_col='featureName')
    print("load internalFeatures csv: ", self.featureConfigCSVFiles[1])
    # list
    self.internalFeatures = list(self.eventsInternalFConfig.index)


    # out features
    self.eventsOut = DotDict()
    # df
    self.eventsOutFConf = pd.read_csv(self.featureConfigCSVFiles[2], index_col='featureName')
    print("load outFeatures csv: ", self.featureConfigCSVFiles[2])
    # list
    self.outFeatures = list(self.eventsOutFConf.index)

    print("Successfully load all .csv files")


def generate_cuda_struct_declaration(self):
    print("--- lgenerate_cuda_struct_declaration ---")

    # mask features: generate struct declaration
    self.codeMaskEventsStruct = generate_struct_declaration(
        ['nev']+self.inMaskFeatures, self.fConf, structName="MaskEvents")
    print("generate cuda struct 'MaskEvents' ")

    # in features: generate struct declaration
    self.codeEventsStruct = generate_struct_declaration(
        ['nev']+self.inSelectionFeatures+self.cumsumedFeatures, self.fConf, structName="Events")
    print("generate cuda struct 'Events' ")


    # internal features: generate struct declaration
    self.codeEventsInternalStruct = generate_struct_declaration(
        self.internalFeatures, self.eventsInternalFConfig, structName="EventsInternal")
    print("generate cuda struct 'EventsInternal' ")


    # out features: generate struct declaration
    self.codeEventsOutStruct = generate_struct_declaration(
        self.outFeatures, self.eventsOutFConf, structName="EventsOut")
    print("generate cuda struct 'EventsOut' ")

    print("Successfully generate all cuda struct ")

    


def compile_cuda_kernels(self):
    print("--- compile_cuda_kernels in .cu files ---")

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
    f = open(self.baseDir + self.cuDir + "/autogen.cu", "w")
    f.write(code)
    f.close()

    # complie cuda code
    module = cu.compiler.SourceModule( code )
    self.kernels = DotDict({ name: module.get_function(name) for name in self.cuKernelsNames })
    
    print("Successfully complie cuda kernels")


    del self.codeMaskEventsStruct 
    del self.codeEventsStruct
    del self.codeEventsInternalStruct
    del self.codeEventsOutStruct



