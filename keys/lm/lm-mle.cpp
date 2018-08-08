#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
double nllCpp(arma::vec pars, arma::vec y, arma::vec x) {

  int n = y.n_elem;       // sample size

  // Parameters. Unlike R, indexing begins with 0
  double beta0 = pars(0);
  double beta1 = pars(1);
  double sig = pars(2);

  arma::vec mu = arma::zeros<arma::vec>(n);

  double nll=0.0;

  for(int i=0; i<n; i++) {
    /* In R, you would compute the likelihoods using a vector operation
	 and then sum. In C++, you add as you go with '+=', or subtract
	 as you go with "-=".
    */
    mu(i) = beta0 + beta1*x(i);
    nll -= R::dnorm(y(i), mu(i), sig, 1);
  }

  return nll;

}
