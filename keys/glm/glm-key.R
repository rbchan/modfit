# Generalized key to evaluating the likelihood equations and joint posterior
# probabilites for a logistic regression (binomial GLM)

# Generate some random data
n <- 100
set.seed(39090)
x <- rnorm(n) # Covariate
beta0 <- -1
beta1 <- 2

# Evaluate the mean and responses from the distributions
eta <- beta0 + beta1*x
psi <- plogis(eta)



plot(x, eta)

plot(x, psi)



y <- rbinom(n, size=1, prob=psi)


plot(x,y)

summary(glm1 <- glm(y~x, family=binomial))



## Evaluate the likelihood of the model

# Create a negative log-likelihood function
nll <- function(pars) {
    beta0 <- pars[1]
    beta1 <- pars[2]
    eta <- beta0 + beta1*x
    psi <- plogis(eta)
    ll <- dbinom(y, size=1, prob=psi, log=TRUE)
    -sum(ll)
}

# Variable inputs for nll function
starts <- c(beta0=0,beta1=0)

nll(starts)


starts2 <- c(beta0=-1,beta1=2)

nll(starts2)




## Minimize negative log-likelihood

## The optim function minimizes the neg log-liklihood
## Hessian = TRUE returns the Hessian matrix (2nd order partial derivatives)
fm <- optim(starts, nll, hessian=TRUE)
fm

# $par has the maximum likelihood estimates
mles <- fm$par

# Obtain the variance-covariance matrix by inverting the Hessian matrix
vcov <- solve(fm$hessian)

# Standard errors taken from the square-root of the diagonals on the variance-covariance matrix
SEs <- sqrt(diag(vcov))

cbind(Est=mles, SE=SEs)

summary(glm1)


#################################################################################
#################################################################################

## Bayesian inference


## Gibbs sampler


# Create a function "lm.gibbs" that uses input data to run n iterations using
# starting values and tuning values
glmB.gibbs <- function(data, niter=10000,
                       start, tune) {

# Create a matrix to store values from iterations
samples <- matrix(NA, niter, 2)
colnames(samples) <- c("beta0", "beta1")

# Store starting values
beta0 <- start[1]
beta1 <- start[2]

psi <- plogis(beta0 + beta1*x)
ll.y <- sum(dbinom(y, size=1, prob=psi, log=TRUE))


for(iter in 1:niter) {
    ### Propose candidates values for each probabilites

    ## Sample from p(beta0|dot)
##    psi <- plogis(beta0 + beta1*x)
##    ll.y <- sum(dbern(y, size=1, prob=psi, log=TRUE))
    # Prior probability of beta0
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    beta0.cand <- rnorm(1, beta0, tune[1])
    psi.cand <- plogis(beta0.cand + beta1*x)
    ll.y.cand <- sum(dbinom(y, size=1, prob=psi.cand, log=TRUE))
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))

    if(runif(1) < mhr) {
        beta0 <- beta0.cand
        psi <- psi.cand
        ll.y <- ll.y.cand
    }

    ## Sample from p(beta1|dot)
    prior.beta1 <- dnorm(beta1, 0, 1000, log=TRUE)

    beta1.cand <- rnorm(1, beta1, tune[2])
    psi.cand <- plogis(beta0 + beta1.cand*x)
    ll.y.cand <- sum(dbinom(y, size=1, prob=psi.cand, log=TRUE))
    prior.beta1.cand <- dnorm(beta1.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta1.cand) - (ll.y+prior.beta1))
    if(runif(1) < mhr) {
        beta1 <- beta1.cand
        psi <- psi.cand
        ll.y <- ll.y.cand
    }


    samples[iter,] <- c(beta0, beta1)


}


return(samples)

}




mc1 <- glmB.gibbs(data=y, niter=10000,
                  start=c(3,-2),
                  tune=c(0.7, 1.2))

str(mc1)

## The MCMC samples from the posterior distribution
plot(mc1)


## Now with arrows indicating the sequence of the first 50 iterations
plot(mc1)
arrows(mc1[1:49,1], mc1[1:49,2], mc1[2:50,1], mc1[2:50,2], length=0.1, col=2)




library(coda)

# Create a Markov chain Monte Carlo object for the
mc1.1 <- mcmc(mc1)

plot(mc1.1)


# Evaluate rejection rate for MH ratio
# Should be somewhere around 60-70 percent
# This is where tuning comes in
rejectionRate(mc1.1)




mc1.1t <- window(mc1.1, start=101, thin=10)

plot(mc1.1t)

summary(mc1.1t)



