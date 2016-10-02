/*
    likelihood.c
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static double log_multi_beta_vector(int *vec, int length, double padding){
    double *new_vec;
    int i;
    double left;
    double right;
    
    if((new_vec = calloc(length,sizeof(double))) == NULL){
        fprintf(stderr,"lda_likelihood:: cannot allocate new_vec.\n");
        exit(1);
    }
    
    for(i = 0;i < length;i++){
        new_vec[i] = (double)vec[i] + padding;
    }
    
    left = 0.0;
    for(i = 0;i < length;i++){
        left += lgamma(new_vec[i]);
    }
    right = 0.0;
    for(i = 0;i < length;i++){
        right += new_vec[i];
    }
    right = lgamma(right);
    
    return (left - right);
}

static double log_multi_beta_scalar(double val, int K){
    return ((double)K * lgamma(val) - lgamma((double)K * val));
}

double loglikelihood(int **n_mz, int **n_zw, int *n_m, int **m_mz, int **m_zt, int *m_z, int nclass, int nlex, int ntlex, int ndocs, int Mall, double alpha, double beta, double gamma, double eta){
    double lik = 0.0;
    int m, k;
    int M0 = m_z[nclass];
    
    // P(W | Z, beta)
    for(k = 0;k < nclass;k++){
        lik += log_multi_beta_vector(n_zw[k], nlex, beta);
        lik -= log_multi_beta_scalar(beta, nlex);
    }
    
    // P(Z | alpha)
    for(m = 0;m < ndocs;m++){
        lik += log_multi_beta_vector(n_mz[m], nclass, alpha);
        lik -= log_multi_beta_scalar(alpha, nclass);
    }
    
    // P(T | C, R, gamma)
    for(k = 0;k < (nclass+1);k++){
        lik += log_multi_beta_vector(m_zt[k], ntlex, gamma);
        lik -= log_multi_beta_scalar(gamma, ntlex);
    }
    
    // P(R | eta)
    lik += (lgamma(2.0 * eta) - 2.0 * lgamma(eta) + lgamma((double)M0 + eta) + lgamma((double)(Mall - M0) + eta) - lgamma((double)Mall + 2.0 * eta));
    
    // P(C | Z)
    for(m = 0;m < ndocs;m++){
        for(k = 0;k < nclass;k++){
            if((m_mz[m][k] ==0) || (n_mz[m][k] == 0))
                continue;
            lik += (double)m_mz[m][k] * (log((double)n_mz[m][k]) - log((double)n_m[m]));
        }
    }
    
    return lik;
}
