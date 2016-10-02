/*
    hyper.h
*/
#ifndef HYPER_H
#define HYPER_H

extern double update_alpha(double alpha, int *n_m, int **n_mz, int ndocs, int nclass);
extern double update_beta(double beta, int *n_z, int **n_zw, int nclass, int nlex);
extern double update_gamma(double gamma, int *m_z, int **m_zt, int nclass, int ntlex);
extern double update_eta(double eta, int M0, int M1);

#endif
