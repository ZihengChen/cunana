
from framework.Utilities import *


class CuHandler():
    def __init__(self, cuDir):
        print("--- handle cu files ---")

        self.set_cu_dir(cuDir)
        self.cuStructs = {}
        self.cuKernels = {}
        self.cuModule = None


    def set_cu_dir(self, cuDir):
        self.cuDir = cuDir
        


    def get_cu_kernels(self, cuKernelsNames):
        print("get cuKernels")
        for name in cuKernelsNames:
            self.cuKernels[name] = self.cuModule.get_function(name)
        self.cuStructs.clear()
        del self.cuModule


    def compile_cu_module(self, cuFiles):
        print("complie cuModule")
        code = ""
        # add struct declarations
        for cuStruct in self.cuStructs.values():
            code += cuStruct

        # read code from src files
        for cuFile in cuFiles:
            print("add cuda code: ", self.cuDir+cuFile)
            f = open(self.cuDir+cuFile, "r") 
            code += f.read()
            f.close()
        
        # save total cu file
        autogenFile = self.cuDir+"autogen.cu"
        f = open(autogenFile, "w")
        f.write(code)
        f.close()
        print("save autogen code to ", autogenFile)

        # compile code
        self.cuModule = cu.compiler.SourceModule(code)
        
        

    def generate_cu_struct(self, cuStructName, fconfLs, fconfDf):
        print("generate cuStruct {}".format(cuStructName))

        # start to generate code
        code = "// {} is auto-generated from csv file \n".format(cuStructName)
        code += "struct " + cuStructName + "{\n"

        # add features as arrays
        for f in fconfLs:
            # use fconfDf to compose
            if f in fconfDf.index:
                fmt = fconfDf.loc[f,'type']
                isArray = fconfDf.loc[f,'isArray']==1
            # cumsum is always array of uint
            elif "cumsum" in f:
                fmt = 'uint'
                isArray = True
            # nev is always a single int
            elif f=="nev":
                fmt = 'int'
                isArray = False
            else:
                raise RuntimeError("cannot generate_struct: type of '{}' is unknown".format(f))
        
            # add * if isArray
            obj = "*"+f if isArray else f
            # convert bool to int because cuda doesn't support array of bool
            if fmt == "bool": fmt = "int"
            # add to code
            code += "    {} {};\n".format(fmt, obj)
        
        # ending parenthesis 
        code += "};\n\n"

        self.cuStructs[cuStructName] = code

