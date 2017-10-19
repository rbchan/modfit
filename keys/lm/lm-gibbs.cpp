#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat lm_gibbs(arma::mat y, arma::mat X, int niter, arma::vec tune) {

  int p = X.n_cols;
  arma::vec beta = arma::randn<arma::vec>(p);
  double sig=0.0;

  arma::vec mu = X*beta;
  arma::vec mu_cand = mu;
  double ll_y = arma::accu(Rcpp::dnorm(y, mu, sig, true));
  double ll_y_cand = ll_y;

  arma::mat samples = arma::zeros<arma::mat>(niter,p+1);

  double mhr=0.0;

  for(int i=0; i<niter; i++) {

    // sample beta0
    beta_cand = beta;
    beta_cand(0) = Rcpp::rnorm(1,beta(0),tune(0));
    mu_cand = X*beta_cand;
    ll_y_cand = arma::accu(Rcpp::dnorm(y, mu_cand, sig, true));
    mhr = exp(ll_y_cand - ll_y);
    if(arma::randu(1) < mhr) {
      beta = beta_cand;
      mu = mu_cand;
      ll_y = ll_y_cand;
    }

    // sample beta1
    beta_cand = beta;
    beta_cand(1) = Rcpp::rnorm(1,beta(1),tune(1));
    mu_cand = X*beta_cand;
    ll_y_cand = arma::accu(Rcpp::dnorm(y, mu_cand, sig, true));
    mhr = exp(ll_y_cand - ll_y);
    if(arma::randu(1) < mhr) {
      beta = beta_cand;
      mu = mu_cand;
      ll_y = ll_y_cand;
    }

    // sample sigma
    sig_cand = Rcpp::rnorm(1,sigma,tune(2));
    ll_y_cand = arma::accu(Rcpp::dnorm(y, mu, sig_cand, true));
    mhr = exp(ll_y_cand - ll_y);
    if(arma::randu(1) < mhr) {
      beta = beta_cand;
      mu = mu_cand;
      ll_y = ll_y_cand;
    }

    samples(i,0) = beta(0);
    samples(i,1) = beta(1);
    samples(i,2) = sig;

  }

  return samples;

}
