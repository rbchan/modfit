# Generalized key to evaluating the likelihood equations and joint posterior
# probabilites for a simple linear model with 2 betas

# Generate some random data
n <- 100
x <- rnorm(n) # Covariate
beta0 <- -1
beta1 <- 1
sigma <- 2

# Evaluate the mean and responses from the distributions
mu <- beta0 + beta1*x
y <- rnorm(n, mu, sigma)


plot(x,y)

summary(lm1 <- lm(y~x))



## Evaluate the likelihood of the model

# Create a negative log-likelihood function
nll <- function(pars) {
    beta0 <- pars[1]
    beta1 <- pars[2]
    sigma <- pars[3]
    mu <- beta0 + beta1*x
    ll <- dnorm(y, mean=mu, sd=sigma, log=TRUE)
    -sum(ll)
}

# Variable inputs for nll function
starts <- c(beta0=0,beta1=0,sigma=1)

nll(starts)


starts2 <- c(beta0=2,beta1=0,sigma=1)

nll(starts2)




## Minimize negative log-likelihood

# The optim function minimizes the neg log-like using the set of inital values to be optimized over.
# Hessian = TRUE returns the Hessian matrix (2nd order partial derivatives)
fm <- optim(starts, nll, hessian=TRUE)
fm

# $par returns the best set of parameters found for the maximum likelihood estimator (mles)
mles <- fm$par

# Obtain the variance-covariance matrix by solving the Hessian matrix
vcov <- solve(fm$hessian)

# Standard errors taken from the square-root of the diagonals on the variance-covariance matrix
SEs <- sqrt(diag(vcov))

cbind(Est=mles, SE=SEs)


###################################################################################################
###################################################################################################

## Bayesian inference


## Gibbs sampler


# Create a function "lm.gibbs" that uses input data to run n iterations using
# starting values and tuning values
lm.gibbs <- function(y, niter=10000,
                     start, tune) {

# Create a matrix to store values from iterations
samples <- matrix(NA, niter, 3)
colnames(samples) <- c("beta0", "beta1", "sigma")

# Store starting values
beta0 <- start[1]
beta1 <- start[2]
sigma <- start[3]

for(iter in 1:niter) {
    ### Propose candidates values for each probabilites

    ## Sample from p(beta0|dot)
    mu <- beta0 + beta1*x
    # Obtain the neg log-likelihood of the reponse variable distribution
    # Assume a normal distribution
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Prior probability of beta0
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    # Propose a candidate value for each beta0 based around the known value of beta0
    # Tuning value allows for exploration of values around beta0 by oscillating values above and below beta0
    # Tuning is a trial and error process and comes into play when assessing rejection rates
    beta0.cand <- rnorm(1, beta0, tune[1])
    # Update mu for the candidate beta0
    mu.cand <- beta0.cand + beta1*x
    # Re-evaluate the summation of log-like of the reponse distribution use the candidate mu
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    # Obtain a new prior probabilty for the candidate beta0
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    # Metropolis-Hastings Ratio is a Markov chain Monte Carlo (MCMC) method used to
    # obtain a sequence of random sample from a probabilty distribution

    # P(data | candidate beta0) * P(candidate beta0) / P(data | beta0) * P(beta0)
    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))

    # If the MH ratio is greater than random deviates of uniform distribution than
    # update beta0 with the new candidate beta0
    if(runif(1) < mhr) {
        beta0 <- beta0.cand
    }

    ## Sample from p(beta1|dot)
    # Similar code instructions as used for beta0
    mu <- beta0 + beta1*x
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    prior.beta1 <- dnorm(beta1, 0, 1000, log=TRUE)

    beta1.cand <- rnorm(1, beta1, tune[2])
    mu.cand <- beta0 + beta1.cand*x
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    prior.beta1.cand <- dnorm(beta1.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta1.cand) - (ll.y+prior.beta1))
    if(runif(1) < mhr) {
        beta1 <- beta1.cand
    }

    ## Sample from p(sigma|dot)
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Assume a uniform distribution
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)

    # ## Symmetric proposal distribution
    # # Should this be runif?
    #  sigma.cand <- rnorm(1, sigma, tune[3])
    #  # Sigma must be postive
    #  if(sigma.cand>0) {
    #      ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    #      prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    #
    #      ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    #      prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    #  }
    #     mhr <- exp((ll.y.cand+prior.sigma.cand) - (ll.y+prior.sigma))

    ## Asymmetric proposal distribution - most commonly due to bounds on parameter values
    # Example: sigma cannot be negative
    # Use the log normal distribution to ensure positive sigma values
    sigma.cand <- rlnorm(1, log(sigma), tune[3])
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Prior probability distribution of sigma
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    # Proportion distribution of sigma
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)

    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    # Prior probability distribution of candidate sigma
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    # Proportion distribution of candidate
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)

    # P(data | candidate sigma) * P(candidate sigma) * P(sigma | candidate sigma) / P(data | sigma) * P(sigma) * P(candidate sigma | sigma)
    mhr <- exp((ll.y.cand+prior.sigma.cand+prop.sigma) -
               (ll.y+prior.sigma+prop.sigma.cand))

    if(runif(1) < mhr) {
        sigma <- sigma.cand
    }

    samples[iter,] <- c(beta0, beta1, sigma)


}


return(samples)

}




mc1 <- lm.gibbs(y=y, niter=1000,
                start=c(0,0,1),
                tune=c(0.1, 0.1, 0.1))

str(mc1)

summary(mc1)

apply(mc1, 2, sd)


plot(mc1[,"beta0"], type="l")
hist(mc1[,"beta0"])



library(coda)

# Create a Markov chain Monte Carlo object for the
mc1.1 <- mcmc(mc1)

# Extract a subset of data between start and end (exclude 1st 100 iterations)
mc1.1t <- window(mc1.1, start=101, thin=1)

summary(mc1.1t)

# Evaluate rejection rate for MH ratio
# Should be somewhere between 20 and 40 percent
# This is where tuning comes in
rejectionRate(mc1.1t)

cbind(Est=mles, SE=SEs)

# Allows for visualization of gibbs sampler
plot(mc1.1t)


###################################################################################################
###################################################################################################

## Bayesian inference with JAGS


library(rjags)
# Create a list of data for jags
jd <- list(y=y, x=x, n=n)
str(jd)

# Define your variable names
jp <- c("beta0", "beta1", "sigma")
jp

# Set initial values
ji <- function() {
    list(beta0=rnorm(1), beta1=rnorm(1), sigmaSq=runif(1))
}

ji()


## Compile the model and adapt
# requires creation of jag file in text editor
jm <- jags.model("lm-JAGS.jag", data=jd, inits=ji, n.chains=3,
                 n.adapt=1000)

jc1 <- coda.samples(jm, jp, n.iter=1000)

plot(jc1)


summary(jc1)

summary(mc1.1t)


###################################################################################################
###################################################################################################

## Faster version of lm.gibbs
## This one avoids redundant likelihood (and other) calculations by updating ll.y and mu

## Gibbs sampler


# Create a function "lm.gibbs" that uses input data to run n iterations using
# starting values and tuning values
lm.gibbs.faster <- function(y, niter=10000,
                            start, tune) {

  # Create a matrix to store values from iterations
  samples <- matrix(NA, niter, 3)
  colnames(samples) <- c("beta0", "beta1", "sigma")

  # Store starting values
  beta0 <- start[1]
  beta1 <- start[2]
  sigma <- start[3]

  # Move all constant variables outside for loop
  mu <- beta0 + beta1*x
  ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))

  for(iter in 1:niter) {
    ### Propose candidates values for each probabilites

    ## Sample from p(beta0|dot)

    # Prior probability of beta0
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    # Propose a candidate value for each beta0 based around the known value of beta0
    # Tuning value allows for exploration of values around beta0 by oscillating values above and below beta0
    # Tuning is a trial and error process and comes into play when assessing rejection rates
    beta0.cand <- rnorm(1, beta0, tune[1])
    # Update mu for the candidate beta0
    mu.cand <- beta0.cand + beta1*x
    # Re-evaluate the summation of log-like of the reponse distribution use the candidate mu
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    # Obtain a new prior probabilty for the candidate beta0
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    # Metropolis-Hastings Ratio is a Markov chain Monte Carlo (MCMC) method used to
    # obtain a sequence of random sample from a probabilty distribution

    # P(data | candidate beta0) * P(candidate beta0) / P(data | beta0) * P(beta0)
    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))

    # If the MH ratio is greater than random deviates of uniform distribution than
    # update beta0 with the new candidate beta0
    # update likelihood with new candidate likelihood
    if(runif(1) < mhr) {
      beta0 <- beta0.cand
      ll.y <- ll.y.cand
      mu <- mu.cand
    }

    ## Sample from p(beta1|dot)
    prior.beta1 <- dnorm(beta1, 0, 1000, log=TRUE)

    beta1.cand <- rnorm(1, beta1, tune[2])
    mu.cand <- beta0 + beta1.cand*x
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    prior.beta1.cand <- dnorm(beta1.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta1.cand) - (ll.y+prior.beta1))
    if(runif(1) < mhr) {
      beta1 <- beta1.cand
      ll.y <- ll.y.cand
      mu <- mu.cand
    }

    ## Sample from p(sigma|dot)

    ## Asymmetric proposal distribution - most commonly due to bounds on parameter values
    # Example: sigma cannot be negative
    # Use the log normal distribution to ensure positive sigma values
    sigma.cand <- rlnorm(1, log(sigma), tune[3])
#    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Prior probability distribution of sigma
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    # Proportion distribution of sigma
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)

    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    # Prior probability distribution of candidate sigma
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    # Proportion distribution of candidate
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)

    # P(data | candidate sigma) * P(candidate sigma) * P(sigma | candidate sigma) / P(data | sigma) * P(sigma) * P(candidate sigma | sigma)
    mhr <- exp((ll.y.cand+prior.sigma.cand+prop.sigma) -
                 (ll.y+prior.sigma+prop.sigma.cand))

    if(runif(1) < mhr) {
      sigma <- sigma.cand
      ll.y <- ll.y.cand
    }

    samples[iter,] <- c(beta0, beta1, sigma)


  }


  return(samples)

}






mc.faster <- lm.gibbs.faster(y=y, niter=1000,
                      start=c(0,0,1),
                      tune=c(0.1, 0.1, 0.1))


plot(mcmc(mc.faster))




























###################################################################################################
###################################################################################################

## Even faster version of lm.gibbs
## This one uses matrix multiplication to compute linear predictor

## Gibbs sampler


# Create a function "lm.gibbs" that uses input data to run n iterations using
# starting values and tuning values
lm.gibbs.evenfaster <- function(y, niter=10000,
                                start, tune) {

  # Create a matrix to store values from iterations
  samples <- matrix(NA, niter, 3)
  colnames(samples) <- c("beta0", "beta1", "sigma")

  # Store starting values
  beta0 <- start[1]
  beta1 <- start[2]
  sigma <- start[3]

  # Move all constant variables outside for loop
##  mu <- beta0 + beta1*x
  X <- model.matrix(~x) ## First column will be 1s for the intercept
  mu <- X %*% c(beta0, beta1)
  ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))

  for(iter in 1:niter) {
    ### Propose candidates values for each probabilites

    ## Sample from p(beta0|dot)

    # Prior probability of beta0
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    # Propose a candidate value for each beta0 based around the known value of beta0
    # Tuning value allows for exploration of values around beta0 by oscillating values above and below beta0
    # Tuning is a trial and error process and comes into play when assessing rejection rates
    beta0.cand <- rnorm(1, beta0, tune[1])
    # Update mu for the candidate beta0
##    mu.cand <- beta0.cand + beta1*x
    mu.cand <- X %*% c(beta0.cand, beta1)

    # Re-evaluate the summation of log-like of the reponse distribution use the candidate mu
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    # Obtain a new prior probabilty for the candidate beta0
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    # Metropolis-Hastings Ratio is a Markov chain Monte Carlo (MCMC) method used to
    # obtain a sequence of random sample from a probabilty distribution

    # P(data | candidate beta0) * P(candidate beta0) / P(data | beta0) * P(beta0)
    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))

    # If the MH ratio is greater than random deviates of uniform distribution than
    # update beta0 with the new candidate beta0
    # update likelihood with new candidate likelihood
    if(runif(1) < mhr) {
      beta0 <- beta0.cand
      ll.y <- ll.y.cand
      mu <- mu.cand
    }

    ## Sample from p(beta1|dot)
    prior.beta1 <- dnorm(beta1, 0, 1000, log=TRUE)

    beta1.cand <- rnorm(1, beta1, tune[2])
##    mu.cand <- beta0 + beta1.cand*x
    mu.cand <- X %*% c(beta0, beta1.cand)
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    prior.beta1.cand <- dnorm(beta1.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta1.cand) - (ll.y+prior.beta1))
    if(runif(1) < mhr) {
      beta1 <- beta1.cand
      ll.y <- ll.y.cand
      mu <- mu.cand
    }

    ## Sample from p(sigma|dot)

    ## Asymmetric proposal distribution - most commonly due to bounds on parameter values
    # Example: sigma cannot be negative
    # Use the log normal distribution to ensure positive sigma values
    sigma.cand <- rlnorm(1, log(sigma), tune[3])
#    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Prior probability distribution of sigma
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    # Proportion distribution of sigma
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)

    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    # Prior probability distribution of candidate sigma
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    # Proportion distribution of candidate
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)

    # P(data | candidate sigma) * P(candidate sigma) * P(sigma | candidate sigma) / P(data | sigma) * P(sigma) * P(candidate sigma | sigma)
    mhr <- exp((ll.y.cand+prior.sigma.cand+prop.sigma) -
                 (ll.y+prior.sigma+prop.sigma.cand))

    if(runif(1) < mhr) {
      sigma <- sigma.cand
      ll.y <- ll.y.cand
    }

    samples[iter,] <- c(beta0, beta1, sigma)


  }


  return(samples)

}






mc.evenfaster <- lm.gibbs.evenfaster(y=y, niter=1000,
                                     start=c(0,0,1),
                                     tune=c(0.1, 0.1, 0.1))


plot(mcmc(mc.evenfaster))
































## Make things very fast with C++


library(Rcpp)
library(RcppArmadillo)

sourceCpp(file="lm-gibbs.cpp")

X <- model.matrix(~x)

mc.fastest <- lm_gibbsCpp(y=y, X=X, niter=1000, tune=c(0.1,0.1,0.1))


plot(mcmc(mc.fastest))

ls()







library(benchmark)



## C++ version is 4-7x faster than R implementations
benchmark(
    lm.gibbs(y=y, niter=10000,start=c(0,0,1), tune=c(0.1,0.1,0.1)),
    lm.gibbs.faster(y=y, niter=10000,start=c(0,0,1), tune=c(0.1,0.1,0.1)),
    lm.gibbs.evenfaster(y=y, niter=10000,start=c(0,0,1), tune=c(0.1,0.1,0.1)),
    lm_gibbsCpp(y=y, X=X, niter=10000, tune=c(0.1,0.1,0.1)),
    columns=c("test", "elapsed", "relative"), replications=10)




## All of this could be made even more efficient by using conguate
## priors and sampling directly from the full conditional
## distributions


