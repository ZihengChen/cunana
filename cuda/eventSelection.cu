


//////////////////
// event selection
//////////////////

__global__ void knl_eventSelection(EventsIn *evsI, EventsMid *evsM, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evsI->nev) {    

        int MAXNLEPTON = evsM->MAXNLEPTON;
        struct P4_PtEtaPhiM lep1, lep2, dilepton;

        evsO->channel[iev] = -1;

        if (evsI->HLT_Ele32_WPTight_Gsf[iev]==1 && evsO->nPassElectron[iev]>=2 && evsO->nPassMuon[iev]==0){
            
            // get index
            int l1 = evsM->iPassElectron[iev*MAXNLEPTON+0] + evsI->cumsum_nElectron[iev];
            int l2 = evsM->iPassElectron[iev*MAXNLEPTON+1] + evsI->cumsum_nElectron[iev];
            
            // pt threshold
            if (evsI->Electron_pt[l1]<32 || evsI->Electron_pt[l2]<20) return;
            

            // opposite sign
            if (evsI->Electron_pdgId[l1] * evsI->Electron_pdgId[l2] > 0) return;

            
            // dilepton mass veto
            lep1 = {evsI->Electron_pt[l1], evsI->Electron_eta[l1], evsI->Electron_phi[l1], evsI->Electron_mass[l1]};
            lep2 = {evsI->Electron_pt[l2], evsI->Electron_eta[l2], evsI->Electron_phi[l2], evsI->Electron_mass[l2]};
            dilepton = lorentz_add(&lep1, &lep2);


            if(dilepton.m<60 || dilepton.m>130) return;
        

            // fillout evsO
            evsO->channel[iev] = 0;
            evsO->lepton1Pdgid[iev] = evsI->Electron_pdgId[l1];
            evsO->lepton2Pdgid[iev] = evsI->Electron_pdgId[l2];
            evsO->lepton1Reliso[iev] = evsI->Electron_pfRelIso03_all[l1];
            evsO->lepton2Reliso[iev] = evsI->Electron_pfRelIso03_all[l2];




        } else if (evsI->HLT_IsoMu24[iev]==1 && evsO->nPassElectron[iev]==0 && evsO->nPassMuon[iev]>=2){
            
            // get index
            int l1 = evsM->iPassMuon[iev*MAXNLEPTON+0] + evsI->cumsum_nMuon[iev];
            int l2 = evsM->iPassMuon[iev*MAXNLEPTON+1] + evsI->cumsum_nMuon[iev];
            
            // pt threshold
            if (evsI->Muon_pt[l1]<27 || evsI->Muon_pt[l2]<10) return;
            

            // opposite sign
            if (evsI->Muon_pdgId[l1] * evsI->Muon_pdgId[l2] > 0) return;
            
            // dilepton mass veto
            lep1 = {evsI->Muon_pt[l1], evsI->Muon_eta[l1], evsI->Muon_phi[l1], evsI->Muon_mass[l1]};
            lep2 = {evsI->Muon_pt[l2], evsI->Muon_eta[l2], evsI->Muon_phi[l2], evsI->Muon_mass[l2]};
            dilepton = lorentz_add(&lep1, &lep2);    


            if(dilepton.m<60 || dilepton.m>130) return;
            
            // fillout evsO
            evsO->channel[iev] = 1;
            evsO->lepton1Pdgid[iev] = evsI->Muon_pdgId[l1];
            evsO->lepton2Pdgid[iev] = evsI->Muon_pdgId[l2];
            evsO->lepton1Reliso[iev] = evsI->Muon_pfRelIso03_all[l1];
            evsO->lepton2Reliso[iev] = evsI->Muon_pfRelIso03_all[l2];


        }



        ///////////////////
        // fill leptons p4
        ///////////////////
        if (evsO->channel[iev] != -1){     
            // lep1 p4
            evsO->lepton1Pt[iev]  = lep1.pt;
            evsO->lepton1Eta[iev] = lep1.eta;
            evsO->lepton1Phi[iev] = lep1.phi;
            evsO->lepton1M[iev]   = lep1.m;
            // lep2 p4
            evsO->lepton2Pt[iev]  = lep2.pt;
            evsO->lepton2Eta[iev] = lep2.eta;
            evsO->lepton2Phi[iev] = lep2.phi;
            evsO->lepton2M[iev]   = lep2.m;
            // dilepton p4
            evsO->dileptonPt[iev] = dilepton.pt;
            evsO->dileptonM[iev]  = dilepton.m;
            // lep1-lep2 delta
            float deltaPhi = phi_mpi_pi(lep1.phi-lep2.phi);
            evsO->leptonsDeltaR[iev]   = sqrt((lep1.eta-lep2.eta)*(lep1.eta-lep2.eta) + deltaPhi*deltaPhi) ;
            evsO->leptonsDeltaPhi[iev] = deltaPhi;

        }

        

    }
}

