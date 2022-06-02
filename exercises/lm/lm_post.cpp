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
  ADREPORT(exp(2*logSigma));
  // log-likelihood
  Type llike = sum(dnorm(y, beta0+beta1*x, exp(logSigma), true));
  // log-prior (should be specified by user)
  Type lprior = dnorm(logSigma, Type(0), Type(2), true) +
    dnorm(beta0, Type(0), Type(10), true) +
    dnorm(beta1, Type(0), Type(10), true);
  // log-posterior (unnormalized)
  Type lpost = llike + lprior;
  return -lpost;
}
