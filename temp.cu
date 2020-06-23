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
    int *Electron_cutBased;
    float *Electron_pfRelIso03_all;
    int *Electron_pdgId;
    float *Muon_pt;
    float *Muon_eta;
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
    int *nPassElectron;
    int *nPassMuon;
};




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
                && -2.5 < evs->Electron_eta[i] < 2.5
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
                && -2.4 < evs->Muon_eta[i] < 2.4
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
