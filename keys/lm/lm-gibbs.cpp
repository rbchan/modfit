#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat lm_gibbsCpp(arma::vec y, arma::mat X, int niter, arma::vec tune) {

  int n = X.n_rows;  // sample size
  int p = X.n_cols;  // number of beta parameters
  arma::vec beta = arma::randn<arma::vec>(p); // initialize beta vector with zeros
  arma::vec beta_cand = beta;
  double sig=0.0;
  double sig_cand=0.0;

  // Unlike R, the "*" operator is for matrix multiplication in RcppArmadillo
  arma::vec mu = X*beta;
  arma::vec mu_cand = mu;
  double ll_y = 0.0;
  for(int i=0; i<n; i++) {
    /* In R, you would compute the likelihoods using a vector operation
	 and then sum. In C++, you add as you go with '+='.
    */
    ll_y += R::dnorm(y(i), mu(i), sig, 1);
  }
  double ll_y_cand = ll_y;

  arma::mat samples = arma::zeros<arma::mat>(niter,p+1);

  double mhr=0.0;

  // In C++, indexing starts at 0, not 1!!!!!!!
  for(int iter=0; iter<niter; iter++) {

    // sample beta0. Implicit flat prior
    beta_cand = beta;
    beta_cand(0) = R::rnorm(beta(0),tune(0));
    mu_cand = X*beta_cand;
    ll_y_cand=0.0; // Important that you reset to 0
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

    // sample beta1. Implicit flat prior
    beta_cand = beta;
    beta_cand(1) = R::rnorm(beta(1),tune(1));
    mu_cand = X*beta_cand;
    ll_y_cand=0.0;  // Here too because it would otherwise continue to accummulate
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

    // sample sigma. Implicit flat prior
    sig_cand = R::rnorm(sig,tune(2));
    if(sig_cand > 0) {
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
    }

    samples(iter,0) = beta(0);
    samples(iter,1) = beta(1);
    samples(iter,2) = sig;

  }

  return samples;

}
