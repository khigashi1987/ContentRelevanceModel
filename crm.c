/*
    crm.c
    Content Relevance Model, main driver.
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "crm.h"
#include "learn.h"
#include "writer.h"
#include "feature.h"
#include "dmatrix.h"
#include "imatrix.h"
#include "util.h"

int main(int argc, char **argv){
    document *data;
    document *tag;
    FILE *pp, *tp, *n_mzp, *n_wzp, *likp, *hyperp; // for phi, theta, n_mz, n_zw, likelihood, hyperparameters
    FILE *sp, *m_mzp, *m_tzp; // for psi, m_mz, m_tz
    char c;
    int nlex, dlenmax;
    int ntlex, tlenmax;
    int ndoc;
    int nclass = NCLASS_DEFAULT;
    int maxiter = MAXITER_DEFAULT;
    double alpha = ALPHA_DEFAULT;
    double beta = BETA_DEFAULT;
    double gamma = GAMMA_DEFAULT;
    double eta = ETA_DEFAULT;
    double **phi;
    double **theta;
    double **psi;
    int **n_mz;
    int **n_zw;
    int **m_mz;
    int **m_zt;
    
    while((c = getopt(argc, argv, "I:K:A:B:G:E:h")) != -1){
        switch(c){
            case 'I': maxiter = atoi(optarg); break;
            case 'K': nclass = atoi(optarg); break;
            case 'A': alpha = atof(optarg); break;
            case 'B': beta = atof(optarg); break;
            case 'G': gamma = atof(optarg); break;
            case 'E': eta = atof(optarg); break;
            case 'h': usage(); break;
            default: usage(); break;
        }
    }
    if(!(argc - optind == 3))
        usage();
    
    // open data
    if((data = feature_matrix(argv[optind], &nlex, &dlenmax, &ndoc)) == NULL){
        fprintf(stderr, "crm:: cannot open training data.\n");
        exit(1);
    }
    
    // open tag data
    if((tag = feature_matrix(argv[optind+1], &ntlex, &tlenmax, &ndoc)) == NULL){
        fprintf(stderr, "crm:: cannot open tag data.\n");
        exit(1);
    }
    
    // open model output
    if(((pp = fopen(strconcat(argv[optind+2], ".phi"),"w")) == NULL)
    || ((tp = fopen(strconcat(argv[optind+2], ".theta"),"w")) == NULL)
    || ((sp = fopen(strconcat(argv[optind+2], ".psi"),"w")) == NULL)
    || ((n_mzp = fopen(strconcat(argv[optind+2], ".n_mz"),"w")) == NULL)
    || ((n_wzp = fopen(strconcat(argv[optind+2], ".n_wz"),"w")) == NULL)
    || ((m_mzp = fopen(strconcat(argv[optind+2], ".m_mz"),"w")) == NULL)
    || ((m_tzp = fopen(strconcat(argv[optind+2], ".m_tz"),"w")) == NULL)
    || ((likp = fopen(strconcat(argv[optind+2], ".lik"),"w")) == NULL)
    || ((hyperp = fopen(strconcat(argv[optind+2], ".hyper"),"w")) == NULL)){
        fprintf(stderr, "crm:: cannot open model outputs.\n");
        exit(1);
    }
    
    // allocate parameters
    if((phi = dmatrix(nlex, nclass)) == NULL){
        fprintf(stderr, "crm:: cannot allocate phi.\n");
        exit(1);
    }
    if((theta = dmatrix(ndoc, nclass)) == NULL){
        fprintf(stderr, "crm:: cannot allocate theta.\n");
        exit(1);
    }
    if((psi = dmatrix(ntlex, (nclass+1))) == NULL){
        fprintf(stderr, "crm:: cannot allocate psi.\n");
        exit(1);
    }
    // n_mz ... number of times document and topic z co-occur
    if((n_mz = imatrix(ndoc, nclass)) == NULL){
        fprintf(stderr, "crm:: cannot allocate n_mz.\n");
        exit(1);
    }
    // n_zw ... number of times topic and word w co-occur
    if((n_zw = imatrix(nclass, nlex)) == NULL){
        fprintf(stderr, "crm:: cannot allocate n_zw.\n");
        exit(1);
    }
    // m_mz ... number of times document and topic z co-occur in "tag set"
    if((m_mz = imatrix(ndoc, (nclass+1))) == NULL){
        fprintf(stderr, "crm:: cannot allocate m_mz.\n");
        exit(1);
    }
    // m_zt ... number of times topics and tag t co-occur. nclass = nclass + 1 (including content-irrelevant topic)
    if((m_zt = imatrix((nclass+1), ntlex)) == NULL){
        fprintf(stderr, "crm:: cannot allocate m_zt.\n");
        exit(1);
    }
    
    crm_learn(data, tag, alpha, beta, gamma, eta, nclass, nlex, ntlex, dlenmax, tlenmax, maxiter, phi, theta, psi, n_mz, n_zw, m_mz, m_zt, likp, hyperp);
    crm_write(pp, tp, sp, n_mzp, n_wzp, m_mzp, m_tzp, phi, theta, psi, n_mz, n_zw, m_mz, m_zt, nclass, nlex, ntlex, ndoc);
    
    free_feature_matrix(data);
    free_feature_matrix(tag);
    free_dmatrix(phi, nlex);
    free_dmatrix(theta, ndoc);
    free_dmatrix(psi, ntlex);
    free_imatrix(n_mz, ndoc);
    free_imatrix(n_zw, nclass);
    free_imatrix(m_mz, ndoc);
    free_imatrix(m_zt, nclass+1);
    
    fclose(pp);
    fclose(tp);
    fclose(sp);
    fclose(n_mzp);
    fclose(n_wzp);
    fclose(m_mzp);
    fclose(m_tzp);
    
    exit(0);
}

void usage(void){
    printf("usage: %s [-I maxiter] [-K n_classes] [-A alpha] [-B beta] [-G gamma] [-E eta] doc tag model\n", "crm");
    exit(0);
}
