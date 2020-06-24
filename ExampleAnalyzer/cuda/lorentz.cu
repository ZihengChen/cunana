
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
