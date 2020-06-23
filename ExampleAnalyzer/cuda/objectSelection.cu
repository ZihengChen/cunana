

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
