
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-06-16 17:01:27
# @modify date 2020-06-16 17:01:27


from Utility import *

def eventSelection_channelFilling_ee(self):

    i1 = self.internal.index_electrons[0]
    i2 = self.internal.index_electrons[1]

    # pt threshold
    if self.event.Electron_pt[i1]<32 or self.event.Electron_pt[i2]<20:
        return

    # opposite signe
    if sign(self.event.Electron_pdgId[i1]) * sign(self.event.Electron_pdgId[i2])>0:
        return

    lep1 = TLorentzVector() 
    lep2 = TLorentzVector()
    lep1.SetPtEtaPhiM(self.event.Electron_pt[i1], self.event.Electron_eta[i1], self.event.Electron_phi[i1], self.event.Electron_mass[i1])
    lep2.SetPtEtaPhiM(self.event.Electron_pt[i2], self.event.Electron_eta[i2], self.event.Electron_phi[i2], self.event.Electron_mass[i2])
    dilepton = lep1+lep2

    # dilepton mass veto
    if dilepton.M()<70 or dilepton.M()>110:
        return

    # fillout
    self.out.Channel = 0
    self.eventSelection_leptonsP4Filling(lep1, lep2)

    self.out.LeptonOne_pdgId = self.event.Electron_pdgId[i1]
    self.out.LeptonTwo_pdgId = self.event.Electron_pdgId[i2]
    self.out.LeptonOne_reliso= self.event.Electron_pfRelIso03_all[i1]
    self.out.LeptonTwo_reliso= self.event.Electron_pfRelIso03_all[i2]



def eventSelection_channelFilling_mumu(self):
    
    i1 = self.internal.index_muons[0]
    i2 = self.internal.index_muons[1]

    # pt threshold
    if self.event.Muon_pt[i1]<32 or self.event.Muon_pt[i2]<20:
        return

    # opposite signe
    if sign(self.event.Muon_pdgId[i1]) * sign(self.event.Muon_pdgId[i2])>0:
        return

    lep1 = TLorentzVector() 
    lep2 = TLorentzVector()
    lep1.SetPtEtaPhiM(self.event.Muon_pt[i1], self.event.Muon_eta[i1], self.event.Muon_phi[i1], self.event.Muon_mass[i1])
    lep2.SetPtEtaPhiM(self.event.Muon_pt[i2], self.event.Muon_eta[i2], self.event.Muon_phi[i2], self.event.Muon_mass[i2])
    dilepton = lep1+lep2

    # dilepton mass veto
    if dilepton.M()<70 or dilepton.M()>110:
        return

    # fillout
    self.out.Channel = 1
    self.eventSelection_leptonsP4Filling(lep1, lep2)

    self.out.LeptonOne_pdgId = self.event.Muon_pdgId[i1]
    self.out.LeptonTwo_pdgId = self.event.Muon_pdgId[i2]
    self.out.LeptonOne_reliso= self.event.Muon_pfRelIso03_all[i1]
    self.out.LeptonTwo_reliso= self.event.Muon_pfRelIso03_all[i2]



def eventSelection_leptonsP4Filling(self, lep1, lep2):
    self.out.LeptonOne_pt    = lep1.Pt()
    self.out.LeptonOne_eta   = lep1.Eta()
    self.out.LeptonOne_phi   = lep1.Phi()
    self.out.LeptonOne_m     = lep1.M()
    
    self.out.LeptonTwo_pt    = lep2.Pt()
    self.out.LeptonTwo_eta   = lep2.Eta()
    self.out.LeptonTwo_phi   = lep2.Phi()
    self.out.LeptonTwo_m     = lep2.M()
    
    self.out.Leptons_deltaR   = lep1.DeltaR(lep2)
    self.out.Leptons_deltaPhi = lep1.DeltaPhi(lep2)

    dilepton = lep1+lep2
    self.out.Dilepton_pt  = dilepton.Pt()
    self.out.Dilepton_eta = dilepton.Eta()
    self.out.Dilepton_phi = dilepton.Phi()
    self.out.Dilepton_m   = dilepton.M()

