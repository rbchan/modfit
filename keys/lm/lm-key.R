## Key with answers to questions in stats/exercises/lm/lm.tex
## Problem involves fitting simple linear model with 2 betas using
## maximum likelihood and MCMC

## Simulate data
n <- 100
x <- rnorm(n) # Covariate
beta0 <- -1
beta1 <- 1
sigma <- 2

mu <- beta0 + beta1*x     # expected value of y
y <- rnorm(n, mu, sigma)  # realized values (the data, ie the response variable)


plot(x,y)


## Fit the model using lm()
summary(lm1 <- lm(y~x))



## Fit the model using maximum likelihood

# Negative log-likelihood function
nll <- function(pars) {
    beta0 <- pars[1]
    beta1 <- pars[2]
    sigma <- pars[3]
    mu <- beta0 + beta1*x
    ll <- dnorm(y, mean=mu, sd=sigma, log=TRUE)
    -sum(ll)
}

# Guess the parameter values and evalueate the likelihood
starts <- c(beta0=0,beta1=0,sigma=1)
nll(starts)


## Another guess. This one is better because nll is lower
starts2 <- c(beta0=-1,beta1=0,sigma=1)
nll(starts2)




## Instead of guessing our way to the maximum likelihood estimates,
## we can minimize the negative log-likelihood using 'optim'

# The optim function minimizes the neg log-like using 'starts' as inital values
# hessian = TRUE returns the Hessian matrix (2nd order partial derivatives)
fm <- optim(starts, nll, hessian=TRUE)
fm

# $par has the maximum likelihood estimator (mles)
mles <- fm$par

# Obtain the variance-covariance matrix by inverting the Hessian matrix
vcov <- solve(fm$hessian)

# Standard errors taken from the square-root of the diagonals of the vcov matrix
SEs <- sqrt(diag(vcov))

## Estimates and SEs
cbind(Est=mles, SE=SEs)




## Take <1s to fit the model with optim, but it we want to speed things up, we
## can write the likelihood in C++ with the help of Rcpp.




library(Rcpp)
library(RcppArmadillo) ## For linear algebra (not used here)

## Compile the C++ code in lm-mle.cpp and create an R function to run it
## The new function is called nllCpp
sourceCpp("lm-mle.cpp")


## Maximize the likelihood written in C++
fm2 <- optim(starts2, nllCpp, hessian=TRUE, y=y, x=x)
fm2



all.equal(fm2, fm2) ## Should be TRUE


## Compare the speed of nll and nllCpp
library(rbenchmark)


## C++ is only about 30% faster in this case
## R function nll is reasonably fast because there aren't any 'for loops'
benchmark(nll(starts), nllCpp(starts, y=y, x=x), replications=1e5)



ls()

















####################################################################################
####################################################################################

## Bayesian inference


## Gibbs sampler


# Create a function "lm.gibbs" that uses input data to run niter iterations using
# starting values and tuning values
lm.gibbs <- function(y, x, niter=10000,
                     start, tune) {

# Create a matrix to store values from iterations
samples <- matrix(NA, niter, 3)
colnames(samples) <- c("beta0", "beta1", "sigma")

# Starting values
beta0 <- start[1]
beta1 <- start[2]
sigma <- start[3]

for(iter in 1:niter) {
    ### Propose candidates values for each parameter

    ## Sample from p(beta0|dot)
    mu <- beta0 + beta1*x
    ## Obtain the log-likelihood of the reponse variable
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Prior probability of beta0
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    ## Propose a candidate value for each beta0 based around the current value of beta0
    ## Tuning value allows for exploration of posterior using a random walk approach
    ## Tuning is a trial and error process. Acceptance rates should be around 30%
    ## Possible to do automatic (adaptive) tuning, but often not worth the extra coding
    beta0.cand <- rnorm(1, beta0, tune[1])
    ## Update mu for the candidate beta0
    mu.cand <- beta0.cand + beta1*x
    ## Re-evaluate the log-like using the candidate mu
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    ## Obtain a new prior probabilty for the candidate beta0
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    ## Metropolis-Hastings algorithm is a MCMC method used to
    ## obtain a sequence of (non-independent) samples from a probabilty distribution

    ## P(data | candidate beta0) * P(candidate beta0) / P(data | beta0) * P(beta0)
    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))

    ## If the MH ratio is greater than U(0,1) then accept beta0.cand
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
    mu <- beta0 + beta1*x
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    # Prior probability density for sigma
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    # Proposal probability density of sigma
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)

    # Likelihood, prior, and proposal for candidate
    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)

    # Metropolis-Hastings ratio
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



## Run the Gibbs sampler
mc1 <- lm.gibbs(y=y, x=x, niter=1000,
                start=c(0,0,1),
                tune=c(0.4, 0.4, 0.2))

str(mc1)


## Posterior summary stats
summary(mc1)
apply(mc1, 2, sd)


## View the results
plot(mc1[,"beta0"], type="l")  # Traceplot
hist(mc1[,"beta0"])            # Histogram



### Use the coda package for better summary/visualization/diagnostic options
library(coda)

## Put the results in a 'mcmc' class
mc1.1 <- mcmc(mc1)


## Traceplots and posterior densities
plot(mc1.1, ask=TRUE)



## Exclude 1st 100 iterations as burn-in
mc1.1t <- window(mc1.1, start=101, thin=1)

summary(mc1.1t)

plot(mc1.1t)



## Compare posteriors to MLEs
cbind(Est=mles, SE=SEs)





## Rejection rate for MH algorithm
## Should be somewhere between 60 and 70 percent
## This is where tuning comes in.
## Increase tuning parameters if rejection rate is too low
rejectionRate(mc1.1t)


######################################################################################
######################################################################################

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
## requires creation of jag file in text editor
jm <- jags.model("lm-JAGS.jag", data=jd, inits=ji, n.chains=3,
                 n.adapt=1000)

jc1 <- coda.samples(jm, jp, n.iter=1000)

plot(jc1)


summary(jc1)

summary(mc1.1t)


######################################################################################
######################################################################################

## Faster version of lm.gibbs
## Avoid redundant likelihood (and other) calculations by updating ll.y and mu

lm.gibbs.faster <- function(y, x, niter=10000,
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

    ## Sample from p(beta0|dot)
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    beta0.cand <- rnorm(1, beta0, tune[1])
    mu.cand <- beta0.cand + beta1*x
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))
    if(runif(1) < mhr) {
      beta0 <- beta0.cand
      ll.y <- ll.y.cand   # Stored to avoid redundant calculation
      mu <- mu.cand       # Stored to avoid redundant calculation
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
      ll.y <- ll.y.cand   # Stored to avoid redundant calculation
      mu <- mu.cand       # Stored to avoid redundant calculation
    }

    ## Sample from p(sigma|dot)
    sigma.cand <- rlnorm(1, log(sigma), tune[3])
#    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)

    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)

    mhr <- exp((ll.y.cand+prior.sigma.cand+prop.sigma) -
                 (ll.y+prior.sigma+prop.sigma.cand))
    if(runif(1) < mhr) {
      sigma <- sigma.cand
      ll.y <- ll.y.cand   # Stored to avoid redundant calculation
    }

    samples[iter,] <- c(beta0, beta1, sigma)


  }


  return(samples)

}






mc.faster <- lm.gibbs.faster(y=y, x=x, niter=1000,
                      start=c(0,0,1),
                      tune=c(0.1, 0.1, 0.1))


plot(mcmc(mc.faster))




























######################################################################################
######################################################################################

## Even faster version of lm.gibbs
## This one uses matrix multiplication to compute linear predictor
lm.gibbs.evenfaster <- function(y, X, niter=10000,
                                start, tune) {

  # Create a matrix to store values from iterations
  samples <- matrix(NA, niter, 3)
  colnames(samples) <- c("beta0", "beta1", "sigma")

  # Store starting values
  beta0 <- start[1]
  beta1 <- start[2]
  sigma <- start[3]

##  mu <- beta0 + beta1*x
  mu <- X %*% c(beta0, beta1)                 # matrix multiplication
  ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))

  for(iter in 1:niter) {

    ## Sample from p(beta0|dot)

    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)
    beta0.cand <- rnorm(1, beta0, tune[1])
##    mu.cand <- beta0.cand + beta1*x
    mu.cand <- X %*% c(beta0.cand, beta1)

    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))
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
    sigma.cand <- rlnorm(1, log(sigma), tune[3])
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)

    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)

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




## Create the design matrix
## Here it's just a matrix 2 columns.
## The first column is all 1s. The second is x
X <- model.matrix(~x)

X %*% c(-1,1)   ## Matrix multiplication returning expected values of y


mc.evenfaster <- lm.gibbs.evenfaster(y=y, X=X, niter=1000,
                                     start=c(0,0,1),
                                     tune=c(0.1, 0.1, 0.1))


plot(mcmc(mc.evenfaster))














## Make things very fast with C++
## Take a look at the C++ code in 'lm-gibbs.cpp'

library(Rcpp)
library(RcppArmadillo)

sourceCpp(file="lm-gibbs.cpp")

X <- model.matrix(~x)

mc.fastest <- lm_gibbsCpp(y=y, X=X, niter=1000, tune=c(0.6,0.6,0.3))


plot(mcmc(mc.fastest))

ls()







library(rbenchmark)



## C++ version is 4-7x faster than R implementations (if you ignore compile time!)
benchmark(
    lm.gibbs(y=y, x=x, niter=10000,start=c(0,0,1), tune=c(0.1,0.1,0.1)),
    lm.gibbs.faster(y=y, x=x, niter=10000,start=c(0,0,1), tune=c(0.1,0.1,0.1)),
    lm.gibbs.evenfaster(y=y, X=X, niter=10000,start=c(0,0,1), tune=c(0.1,0.1,0.1)),
    lm_gibbsCpp(y=y, X=X, niter=10000, tune=c(0.1,0.1,0.1)),
    columns=c("test", "elapsed", "relative"), replications=10)




## All of this could be made even more efficient by using conguate
## priors and sampling directly from the full conditional
## distributions (instead of using MH algorithm)











## Running multiple chains in parallel




library(parallel)


## Use up to 3 cores
nCores <- min(detectCores()-1, 3)


## Make a cluster
cl <- makeCluster(nCores)


## Send objects to each core
clusterExport(cl, c("lm.gibbs", "lm_gibbsCpp", "y", "x", "X"))


## Set RNG seeds
clusterSetRNGStream(cl, 3479)


## Run code in {} on each core
## Won't work with C++ function unless you compile it on each core. See below.
out <- clusterEvalQ(cl, {
##    mc <- lm_gibbsCpp(y=y, X=X, niter=10000, tune=c(0.1,0.1,0.1))
    mc <- lm.gibbs(y=y, x=x, niter=1000,
                   start=c(0,0,1), tune=c(0.7,0.7,0.3))
    library(coda)
    return(mcmc(mc))
})


library(coda)
mcl <- as.mcmc.list(out)

plot(mcl)








## Using C++ version
## Compile time makes this option slow. There must be a better way!
out <- clusterEvalQ(cl, {
    library(Rcpp)
    sourceCpp("lm-gibbs.cpp")
    mc <- lm_gibbsCpp(y=y, X=X, niter=10000, tune=c(0.6,0.6,0.3))
    library(coda)
    return(mcmc(mc))
})


library(coda)
mcl <- as.mcmc.list(out)








stopCluster(cl) # good practice
