
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-06-16 17:01:35
# @modify date 2020-06-16 17:01:35

from Utility import *

def objectSelection_electrons(self):
    self.internal.index_electrons = []

    for i in range(self.event.nElectron):
    
        if self.event.Electron_pt[i]>20 \
        and abs(self.event.Electron_eta[i])<2.5 \
        and self.event.Electron_cutBased[i]>=3  \
        :
            self.internal.index_electrons.append(i)
        
    self.out.nElectrons = len(self.internal.index_electrons)


def objectSelection_muons(self):
    self.internal.index_muons = []

    for i in range(self.event.nMuon):
    
        if self.event.Muon_pt[i]>10             \
        and abs(self.event.Muon_eta[i])<2.4     \
        and self.event.Muon_isGlobal            \
        and self.event.Muon_isPFcand            \
        and self.event.Muon_tightId             \
        :
            self.internal.index_muons.append(i)
        
    self.out.nMuons = len(self.internal.index_muons)