#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat lm_gibbsCpp(arma::vec y, arma::mat X, int niter, arma::vec tune) {

  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec beta = arma::randn<arma::vec>(p);
  arma::vec beta_cand = beta;
  double sig=0.0;
  double sig_cand=0.0;

  arma::vec mu = X*beta;
  arma::vec mu_cand = mu;
  double ll_y = 0.0;
  for(int i=0; i<n; i++) {
    ll_y += R::dnorm(y(i), mu(i), sig, 1);
  }
  double ll_y_cand = ll_y;

  arma::mat samples = arma::zeros<arma::mat>(niter,p+1);

  double mhr=0.0;

  for(int iter=0; iter<niter; iter++) {

    // sample beta0
    beta_cand = beta;
    beta_cand(0) = R::rnorm(beta(0),tune(0));
    mu_cand = X*beta_cand;
    ll_y_cand=0.0;
    for(int i=0; i<n; i++) {
      ll_y_cand += R::dnorm(y(i), mu_cand(i), sig, 1);
    }
    mhr = exp(ll_y_cand - ll_y);
    double u = R::runif(0,1);
    if(u < mhr) {
      beta = beta_cand;
      mu = mu_cand;
      ll_y = ll_y_cand;
    }

    // sample beta1
    beta_cand = beta;
    beta_cand(1) = R::rnorm(beta(1),tune(1));
    mu_cand = X*beta_cand;
    ll_y_cand=0.0;
    for(int i=0; i<n; i++) {
      ll_y_cand += R::dnorm(y(i), mu_cand(i), sig, 1);
    }
    mhr = exp(ll_y_cand - ll_y);
    u = R::runif(0,1);
    if(u < mhr) {
      beta = beta_cand;
      mu = mu_cand;
      ll_y = ll_y_cand;
    }

    // sample sigma
    sig_cand = R::rnorm(sig,tune(2));
    ll_y_cand=0.0;
    for(int i=0; i<n; i++) {
      ll_y_cand += R::dnorm(y(i), mu(i), sig_cand, 1);
    }
    mhr = exp(ll_y_cand - ll_y);
    u = R::runif(0,1);
    if(u < mhr) {
      sig = sig_cand;
      mu = mu_cand;
      ll_y = ll_y_cand;
    }

    samples(iter,0) = beta(0);
    samples(iter,1) = beta(1);
    samples(iter,2) = sig;

  }

  return samples;

}
