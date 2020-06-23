


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
