//////////////////
// event selection
//////////////////


__global__ void knl_eventSelection(Events *evs, EventsInternal *evsI, EventsOut *evsO) {
    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evs->nev) {    

        int channel = -1;

        if (evs->HLT_Ele32_WPTight_Gsf[iev]==1 && evsO->nPassElectron>=2 && evsO->nPassMuons==0){
            int l1 = evsI->iPassElectron[iev*evsI->MAXNLEPTON+0] + evs->cumsum_nElectron[iev];
            int l2 = evsI->iPassElectron[iev*evsI->MAXNLEPTON+1] + evs->cumsum_nElectron[iev];

            if (evs->Electron_pt[l1]<32 || evs->Electron_pt[l2]<20){
                return;
            }


            if (evs->Electron_pdgId[l1] * evs->Electron_pdgId[l2] > 0){
                return;
            }




        } else if (evs->HLT_IsoMu24[iev]==1 && evsO->nPassElectron==0 && evsO->nPassMuons>=2){

        }

        

    }
}