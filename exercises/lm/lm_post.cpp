// Simple linear regression.
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(y);
  DATA_VECTOR(x);
  PARAMETER(beta0);
  PARAMETER(beta1);
  PARAMETER(logSigma);
  // ADREPORT(exp(2*logSigma));
  Type ll = sum(dnorm(y, beta0+beta1*x, exp(logSigma), true));
  Type lp = ll +
    dnorm(beta0, Type(0), Type(10), true) +  // Prior on beta0
    dnorm(beta1, Type(0), Type(10), true) +  // Prior on beta1
    dnorm(logSigma, Type(0), Type(2), true); // Prior on log(sigma)
  return lp;
}
