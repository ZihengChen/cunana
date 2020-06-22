
#define MAXNLEPTON 5

__global__ void knl_objectSelection_electrons(Events *evs) {

    int iev = blockDim.x*blockIdx.x + threadIdx.x;

    if (iev < evs->nev) {
        
        // electrons
        
        
        const int cumsum_nElectron = evs->cumsum_nElectron[iev];
        const int nElectron = evs->nElectron[iev]; 

        int iPassElectron[MAXNLEPTON];
        int nPassElectron = 0;

        // loop over all enectrons in the event
        for( int i = cumsum_nElectron; i < cumsum_nElectron+nElectron; i++){
            if (nPassElectron >= MAXNLEPTON) break;
            
            if( evs->Electron_pt[i] > 20
                && -2.5 < evs->Electron_eta[i] < 2.5
                && evs->Electron_cutBased[i] >= 3
            ){
                iPassElectron[nPassElectron] = i-;
                nPassElectron++;
            }
        } // end of loop

    }
}
