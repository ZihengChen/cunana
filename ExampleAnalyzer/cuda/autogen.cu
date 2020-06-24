// MaskEvents is auto-generated from csv file 
struct MaskEvents{
    int nev;
    uint *luminosityBlock;
    int *HLT_Ele32_WPTight_Gsf;
    int *HLT_IsoMu24;
    uint *nElectron;
    uint *nMuon;
};

// Events is auto-generated from csv file 
struct Events{
    int nev;
    int *HLT_Ele32_WPTight_Gsf;
    int *HLT_IsoMu24;
    uint *nElectron;
    uint *nMuon;
    float *Electron_pt;
    float *Electron_eta;
    float *Electron_phi;
    float *Electron_mass;
    int *Electron_cutBased;
    float *Electron_pfRelIso03_all;
    int *Electron_pdgId;
    float *Muon_pt;
    float *Muon_eta;
    float *Muon_phi;
    float *Muon_mass;
    int *Muon_isGlobal;
    int *Muon_isPFcand;
    int *Muon_tightId;
    float *Muon_pfRelIso03_all;
    int *Muon_pdgId;
    uint *cumsum_nElectron;
    uint *cumsum_nMuon;
};

// EventsInternal is auto-generated from csv file 
struct EventsInternal{
    int MAXNLEPTON;
    int *iPassElectron;
    uint *iPassMuon;
};

// EventsOut is auto-generated from csv file 
struct EventsOut{
    int *channel;
    int *nPassElectron;
    int *nPassMuon;
    float *lepton1Pt;
    float *lepton1Eta;
    float *lepton1Phi;
    float *lepton1M;
    float *lepton2Pt;
    float *lepton2Eta;
    float *lepton2Phi;
    float *lepton2M;
    float *dileptonPt;
    float *dileptonM;
    float *leptonsDeltaPhi;
    float *leptonsDeltaR;
    float *lepton1Pdgid;
    float *lepton2Pdgid;
    float *lepton1Reliso;
    float *lepton2Reliso;
};


// some handy lorentz verctor and methords

struct P4_PtEtaPhiM{
    float pt;
    float eta;
    float phi;
    float m;
};

__device__ P4_PtEtaPhiM lorentz_add( P4_PtEtaPhiM *p1, P4_PtEtaPhiM *p2){
    float px1 = p1->pt*cos(p1->phi);
    float py1 = p1->pt*sin(p1->phi);
    float pz1 = p1->pt*sinh(p1->eta);
    float pe1 = sqrt(px1*px1 + py1*py1 + pz1*pz1 + p1->m*p1->m);

    float px2 = p2->pt*cos(p2->phi);
    float py2 = p2->pt*sin(p2->phi);
    float pz2 = p2->pt*sinh(p2->eta);
    float pe2 = sqrt(px2*px2 + py2*py2 + pz2*pz2 + p2->m*p2->m);

    float qx = px1+px2;
    float qy = py1+py2;
    float qz = pz1+pz2;
    float qe = pe1+pe2;

    float q_pt = sqrt(qx*qx + qy*qy);
    float q_eta = 0.0; // FIX ME
    float q_phi = 0.0; // FIX ME
    float q_m  = sqrt(qe*qe - qx*qx - qy*qy - qz*qz);

    struct P4_PtEtaPhiM q = {q_pt, q_eta, q_phi, q_m};
    return q;
}

// #define M_PI 3.1415926
// root function return phi in [-pi,pi]
//https://root.cern.ch/doc/master/TVector2_8cxx_source.html#l00103
__device__ float phi_mpi_pi(float x){
    while(x>M_PI)  x -= 2*M_PI;
    while(x<-M_PI) x += 2*M_PI;
    return x;
}



//////////////
//   mask   //
//////////////

__global__ void knl_mask(MaskEvents *evs, bool *mask) {

    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evs->nev) {  
        bool isPass = false;
        if ( (evs->HLT_Ele32_WPTight_Gsf[iev] || evs->HLT_IsoMu24[iev]) 
            && (evs->nElectron[iev]>=2 || evs->nMuon[iev]>=2)
            ){
                isPass = true;
            }
        mask[iev] = isPass;
    }
}


//////////////////
// obj-electron //
//////////////////
__global__ void knl_objectSelection_electron(Events *evs, EventsInternal *evsI, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evs->nev) {    

        const int cumsum_nObject = evs->cumsum_nElectron[iev];
        const int nObject = evs->nElectron[iev]; 
        
        int nPassObject = 0;
        // loop over all enectrons in the event
        for( int i = cumsum_nObject; i < cumsum_nObject + nObject; i++){
            if (nPassObject >= evsI->MAXNLEPTON) break;
            
            if( evs->Electron_pt[i] > 20
                && abs(evs->Electron_eta[i]) < 2.5
                && evs->Electron_cutBased[i] >= 3
            ){
                evsI->iPassElectron[iev*evsI->MAXNLEPTON + nPassObject] = i-cumsum_nObject;
                nPassObject++;
            }
        } // end of loop
        evsO->nPassElectron[iev] = nPassObject;
    }
}



//////////////////
//    obj-muon  //
//////////////////
__global__ void knl_objectSelection_muon(Events *evs, EventsInternal *evsI, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evs->nev) {        
        
        const int cumsum_nObject = evs->cumsum_nMuon[iev];
        const int nObject = evs->nMuon[iev]; 

        int nPassObject = 0;

        // loop over all enectrons in the event
        for( int i = cumsum_nObject; i < cumsum_nObject + nObject; i++){
            if (nPassObject >= evsI->MAXNLEPTON) break;
            
            if( evs->Muon_pt[i] > 10
                && abs(evs->Muon_eta[i]) < 2.4
                && evs->Muon_isGlobal[i] == 1
                && evs->Muon_isPFcand[i] == 1
                && evs->Muon_tightId[i]  == 1
            ){
                evsI->iPassMuon[iev*evsI->MAXNLEPTON + nPassObject] = i-cumsum_nObject;
                nPassObject++;
            }
        } // end of loop
        evsO->nPassMuon[iev] = nPassObject;
    }
}


//////////////////
// event selection
//////////////////

__global__ void knl_eventSelection(Events *evs, EventsInternal *evsI, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evs->nev) {    

        int MAXNLEPTON = evsI->MAXNLEPTON;
        struct P4_PtEtaPhiM lep1, lep2, dilepton;

        evsO->channel[iev] = -1;

        if (evs->HLT_Ele32_WPTight_Gsf[iev]==1 && evsO->nPassElectron[iev]>=2 && evsO->nPassMuon[iev]==0){
            
            // get index
            int l1 = evsI->iPassElectron[iev*MAXNLEPTON+0] + evs->cumsum_nElectron[iev];
            int l2 = evsI->iPassElectron[iev*MAXNLEPTON+1] + evs->cumsum_nElectron[iev];
            
            // pt threshold
            if (evs->Electron_pt[l1]<32 || evs->Electron_pt[l2]<20) return;
            

            // opposite sign
            if (evs->Electron_pdgId[l1] * evs->Electron_pdgId[l2] > 0) return;

            
            // dilepton mass veto
            lep1 = {evs->Electron_pt[l1], evs->Electron_eta[l1], evs->Electron_phi[l1], evs->Electron_mass[l1]};
            lep2 = {evs->Electron_pt[l2], evs->Electron_eta[l2], evs->Electron_phi[l2], evs->Electron_mass[l2]};
            dilepton = lorentz_add(&lep1, &lep2);


            if(dilepton.m<60 || dilepton.m>130) return;
        

            // fillout evsO
            evsO->channel[iev] = 0;
            evsO->lepton1Pdgid[iev] = evs->Electron_pdgId[l1];
            evsO->lepton2Pdgid[iev] = evs->Electron_pdgId[l2];
            evsO->lepton1Reliso[iev] = evs->Electron_pfRelIso03_all[l1];
            evsO->lepton2Reliso[iev] = evs->Electron_pfRelIso03_all[l2];




        } else if (evs->HLT_IsoMu24[iev]==1 && evsO->nPassElectron[iev]==0 && evsO->nPassMuon[iev]>=2){
            
            // get index
            int l1 = evsI->iPassMuon[iev*MAXNLEPTON+0] + evs->cumsum_nMuon[iev];
            int l2 = evsI->iPassMuon[iev*MAXNLEPTON+1] + evs->cumsum_nMuon[iev];
            
            // pt threshold
            if (evs->Muon_pt[l1]<27 || evs->Muon_pt[l2]<10) return;
            

            // opposite sign
            if (evs->Muon_pdgId[l1] * evs->Muon_pdgId[l2] > 0) return;
            
            // dilepton mass veto
            lep1 = {evs->Muon_pt[l1], evs->Muon_eta[l1], evs->Muon_phi[l1], evs->Muon_mass[l1]};
            lep2 = {evs->Muon_pt[l2], evs->Muon_eta[l2], evs->Muon_phi[l2], evs->Muon_mass[l2]};
            dilepton = lorentz_add(&lep1, &lep2);    


            if(dilepton.m<60 || dilepton.m>130) return;
            
            // fillout evsO
            evsO->channel[iev] = 1;
            evsO->lepton1Pdgid[iev] = evs->Muon_pdgId[l1];
            evsO->lepton2Pdgid[iev] = evs->Muon_pdgId[l2];
            evsO->lepton1Reliso[iev] = evs->Muon_pfRelIso03_all[l1];
            evsO->lepton2Reliso[iev] = evs->Muon_pfRelIso03_all[l2];


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