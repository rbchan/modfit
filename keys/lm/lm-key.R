
n <- 100
x <- rnorm(n) # Covariate
beta0 <- -1
beta1 <- 1
sigma <- 2

mu <- beta0 + beta1*x
y <- rnorm(n, mu, sigma)


plot(x,y)

summary(lm1 <- lm(y~x))



## Likelihood

nll <- function(pars) {
    beta0 <- pars[1]
    beta1 <- pars[2]
    sigma <- pars[3]
    mu <- beta0 + beta1*x
    ll <- dnorm(y, mean=mu, sd=sigma, log=TRUE)
    -sum(ll)
}

starts <- c(beta0=0,beta1=0,sigma=1)

nll(starts)


starts2 <- c(beta0=2,beta1=0,sigma=1)

nll(starts2)




## Minimize neg log-like

fm <- optim(starts, nll, hessian=TRUE)
fm

mles <- fm$par

vcov <- solve(fm$hessian)

SEs <- sqrt(diag(vcov))

cbind(Est=mles, SE=SEs)





## Bayesian inference


## Gibbs sampler



lm.gibbs <- function(data, niter=10000,
                     start, tune) {

samples <- matrix(NA, niter, 3)
colnames(samples) <- c("beta0", "beta1", "sigma")

beta0 <- start[1]
beta1 <- start[2]
sigma <- start[3]

for(iter in 1:niter) {
    ## Sample from p(beta0|dot)
    mu <- beta0 + beta1*x
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    prior.beta0 <- dnorm(beta0, 0, 1000, log=TRUE)

    beta0.cand <- rnorm(1, beta0, tune[1])
    mu.cand <- beta0.cand + beta1*x
    ll.y.cand <- sum(dnorm(y, mu.cand, sigma, log=TRUE))
    prior.beta0.cand <- dnorm(beta0.cand, 0, 1000, log=TRUE)

    mhr <- exp((ll.y.cand+prior.beta0.cand) - (ll.y+prior.beta0))
    if(runif(1) < mhr) {
        beta0 <- beta0.cand
    }

    ## Sample from p(beta1|dot)
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
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)

    sigma.cand <- rnorm(1, sigma, tune[3])
    if(sigma.cand>0) {
        ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
        prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)

        ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
        prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    }

    mhr <- exp((ll.y.cand+prior.sigma.cand) - (ll.y+prior.sigma))
    if(runif(1) < mhr) {
        sigma <- sigma.cand
    }

    samples[iter,] <- c(beta0, beta1, sigma)


}


return(samples)


}



mc1 <- lm.gibbs(data=y, niter=1000,
                start=c(0,0,1),
                tune=c(0.1, 0.1, 0.1))


plot(mc1[,"beta0"], type="l")
hist(mc1[,"beta0"])



library(coda)


plot(mcmc(mc1))
