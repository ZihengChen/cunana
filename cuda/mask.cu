


//////////////
//   mask   //
//////////////

__global__ void knl_mask(MaskEventsIn *evsI, bool *mask) {

    int iev = blockDim.x*blockIdx.x + threadIdx.x;
    if (iev < evsI->nev) {  
        bool isPass = false;
        if ( (evsI->HLT_Ele32_WPTight_Gsf[iev] || evsI->HLT_IsoMu24[iev]) 
            && (evsI->nElectron[iev]>=2 || evsI->nMuon[iev]>=2)
            ){
                isPass = true;
            }
        mask[iev] = isPass;
    }
}
