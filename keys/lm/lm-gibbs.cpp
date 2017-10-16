#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericMatrix distmat(Rcpp::NumericMatrix s, Rcpp::NumericMatrix x) {

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
