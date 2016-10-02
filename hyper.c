/*
    hyper.c
*/
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_sf_psi.h>

static double update_hyper(double param, int *vec, int **mat, int nrow, int ncol){
    double numer = 0.0;
    double denom = 0.0;
    int i, j;
    
    for(i = 0;i < nrow;i++)
        for(j = 0;j < ncol;j++)
            numer += gsl_sf_psi((double)mat[i][j] + param);
    numer -= (double)nrow * (double)ncol * gsl_sf_psi(param);
    
    for(i = 0;i < nrow;i++)
        denom += (double)ncol * gsl_sf_psi((double)vec[i] + param * (double)ncol);
    denom -= (double)nrow * (double)ncol * gsl_sf_psi(param * (double)ncol);
    
    return param * numer / denom;
}

double update_alpha(double alpha, int *n_m, int **n_mz, int ndocs, int nclass){
    double new_alpha = update_hyper(alpha, n_m, n_mz, ndocs, nclass);
    return new_alpha;
}

double update_beta(double beta, int *n_z, int **n_zw, int nclass, int nlex){
    double new_beta = update_hyper(beta, n_z, n_zw, nclass, nlex);
    return new_beta;
}

double update_gamma(double gamma, int *m_z, int **m_zt, int nclass, int ntlex){
    double new_gamma = update_hyper(gamma, m_z, m_zt, nclass, ntlex);
    return new_gamma;
}

double update_eta(double eta, int M0, int M1){
    double numer = 0.0;
    double denom = 0.0;
    
    numer += gsl_sf_psi((double)M0 + eta);
    numer += gsl_sf_psi((double)M1 + eta);
    numer -= 2.0 * gsl_sf_psi(eta);
    
    denom += 2.0 * gsl_sf_psi((double)M0 + (double)M1 + 2.0 * eta);
    denom -= 2.0 * gsl_sf_psi(2.0 * eta);
    
    return eta * numer / denom;
}
