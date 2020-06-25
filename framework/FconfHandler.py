import pandas as pd


class FconfHandler():

    def __init__(self, fconfDir):

        self.set_fconf_dir(fconfDir)

        print("--- handle featureConfig files ---")
        self.handle_fconfin()
        self.handle_fconfmid()
        self.handle_fconfout()


    def set_fconf_dir(self, fconfDir):
        self.fconfIn  = fconfDir + "in.csv"
        self.fconfMid = fconfDir + "mid.csv"
        self.fconfOut = fconfDir + "out.csv"


    def handle_fconfin(self):
        print("load fconfIn: "+ self.fconfIn)
        # load fconfDf
        self.fconfInDf = pd.read_csv(self.fconfIn, index_col="featureName")
        # prepare fconfLs
        self.fconfInLs = list(self.fconfInDf.index)
        # other useful fconfInLs
        self.fconfInLsNeedSave     = list(self.fconfInDf.query("needSave==1").index)
        self.fconfInLsNeedCumsum   = list(self.fconfInDf.query("needCumsum==1").index)
        self.fconfInLsInMask       = list(self.fconfInDf.query("inMask==1").index)
        self.fconfInLsInSelection  = list(self.fconfInDf.query("inSelection==1").index)

        # used for dSoA and cuStructs
        self.fconfInLsMaskEventsIn = ['nev']+self.fconfInLsInMask
        self.fconfInLsEventsIn     = ['nev']+self.fconfInLsInSelection+["cumsum_"+k for k in self.fconfInLsNeedCumsum]
        
    

    def handle_fconfmid(self):
        print("load fconfMid: "+ self.fconfMid)
        # load fconfDf
        self.fconfMidDf = pd.read_csv(self.fconfMid, index_col='featureName')
        # prepare fconfLs
        self.fconfMidLs = list(self.fconfMidDf.index)


    def handle_fconfout(self):
        print("load fconfOut: "+ self.fconfOut)
        # load fconfDf
        self.fconfOutDf = pd.read_csv(self.fconfOut, index_col='featureName')
        # prepare fconfLs
        self.fconfOutLs = list(self.fconfOutDf.index)





