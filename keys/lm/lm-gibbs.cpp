#include <Rcpp.h>

// [[Rcpp::export]]
arma::mat distmat(arma::mat y, arma::mat X, int niter, arma::vec tune) {

  int p = X.n_cols;
  arma::vec beta = arma::randn<arma::vec>(p);
  arma::double sig=0.0;

  arma::vec mu = X*beta;
  arma::vec ll_y = sum(Rcpp::dnorm(y, mu, sig, true))

  for(int i=0; i<niter, i++) {
    // sample beta0
    beta_cand = beta;
    beta_cand(0) = Rcpp::rnorm(1,beta(0),tune(0));


  }


  int M = s.nrow();
  int J = x.nrow();
  Rcpp::NumericMatrix dist(M,J);
  for(int i=0; i<M; i++) {
    for(int j=0; j<J; j++) {
      dist(i,j) = sqrt(pow(s(i,0)-x(j,0), 2) + pow(s(i,1)-x(j,1), 2));
    }
  }
  return dist;

}
