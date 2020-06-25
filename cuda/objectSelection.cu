


//////////////////
// obj-electron //
//////////////////
__global__ void knl_objectSelection_electron(EventsIn *evsI, EventsMid *evsM, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evsI->nev) {    

        const int cumsum_nObject = evsI->cumsum_nElectron[iev];
        const int nObject = evsI->nElectron[iev]; 
        
        int nPassObject = 0;
        // loop over all enectrons in the event
        for( int i = cumsum_nObject; i < cumsum_nObject + nObject; i++){
            if (nPassObject >= evsM->MAXNLEPTON) break;
            
            if( evsI->Electron_pt[i] > 20
                && abs(evsI->Electron_eta[i]) < 2.5
                && evsI->Electron_cutBased[i] >= 3
            ){
                evsM->iPassElectron[iev*evsM->MAXNLEPTON + nPassObject] = i-cumsum_nObject;
                nPassObject++;
            }
        } // end of loop
        evsO->nPassElectron[iev] = nPassObject;
    }
}



//////////////////
//    obj-muon  //
//////////////////
__global__ void knl_objectSelection_muon(EventsIn *evsI, EventsMid *evsM, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evsI->nev) {        
        
        const int cumsum_nObject = evsI->cumsum_nMuon[iev];
        const int nObject = evsI->nMuon[iev]; 

        int nPassObject = 0;

        // loop over all enectrons in the event
        for( int i = cumsum_nObject; i < cumsum_nObject + nObject; i++){
            if (nPassObject >= evsM->MAXNLEPTON) break;
            
            if( evsI->Muon_pt[i] > 10
                && abs(evsI->Muon_eta[i]) < 2.4
                && evsI->Muon_isGlobal[i] == 1
                && evsI->Muon_isPFcand[i] == 1
                && evsI->Muon_tightId[i]  == 1
            ){
                evsM->iPassMuon[iev*evsM->MAXNLEPTON + nPassObject] = i-cumsum_nObject;
                nPassObject++;
            }
        } // end of loop
        evsO->nPassMuon[iev] = nPassObject;
    }
}

