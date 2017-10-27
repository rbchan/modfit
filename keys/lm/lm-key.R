## ------------------------------------------------------------------------
set.seed(348720) # To make this reproducible
n <- 100
x <- rnorm(n) # Covariate
beta0 <- -1
beta1 <- 1
sigma <- 2

mu <- beta0 + beta1*x     # expected value of y
y <- rnorm(n, mu, sigma)  # realized values (ie, the response variable)

## ----scat----------------------------------------------------------------
cbind(x,y)[1:4,] # First 4 observations
plot(x,y)

## ------------------------------------------------------------------------
nll <- function(pars) {
    beta0 <- pars[1]
    beta1 <- pars[2]
    sigma <- pars[3]
    mu <- beta0 + beta1*x
    ll <- dnorm(y, mean=mu, sd=sigma, log=TRUE)
    -sum(ll)
}

## ------------------------------------------------------------------------
# Guess the parameter values and evalueate the likelihood
starts <- c(beta0=0,beta1=0,sigma=1)
nll(starts)

## Another guess. This one is better because nll is lower
starts2 <- c(beta0=-1,beta1=0,sigma=1)
nll(starts2)

## ------------------------------------------------------------------------
fm <- optim(starts, nll, hessian=TRUE)
fm

## ------------------------------------------------------------------------
vcov <- solve(fm$hessian)
SEs <- sqrt(diag(vcov))

## ------------------------------------------------------------------------
mles <- fm$par # The maximum likelihood estimates
cbind(Est=mles, SE=SEs)

## ------------------------------------------------------------------------
summary(lm(y~x))

## ------------------------------------------------------------------------
lm.gibbs <- function(y, x, niter=10000, start, tune) {
samples <- matrix(NA, niter, 3)
colnames(samples) <- c("beta0", "beta1", "sigma")
beta0 <- start[1]; beta1 <- start[2]; sigma <- start[3]

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
    sigma.cand <- rlnorm(1, log(sigma), tune[3])
    mu <- beta0 + beta1*x
    ll.y <- sum(dnorm(y, mu, sigma, log=TRUE))
    prior.sigma <- dunif(sigma, 0, 1000, log=TRUE)
    prop.sigma <- dlnorm(sigma, log(sigma.cand), tune[3], log=TRUE)
    ll.y.cand <- sum(dnorm(y, mu, sigma.cand, log=TRUE))
    prior.sigma.cand <- dunif(sigma.cand, 0, 1000, log=TRUE)
    prop.sigma.cand <- dlnorm(sigma.cand, log(sigma), tune[3], log=TRUE)
    mhr <- exp((ll.y.cand+prior.sigma.cand+prop.sigma) -
               (ll.y+prior.sigma+prop.sigma.cand))
    if(runif(1) < mhr) {
        sigma <- sigma.cand
    }
    samples[iter,] <- c(beta0, beta1, sigma)
}
return(samples)
}

## ------------------------------------------------------------------------
out1 <- lm.gibbs(y=y, x=x, niter=1000,
                start=c(0,0,1),
                tune=c(0.4, 0.4, 0.2))

## ------------------------------------------------------------------------
library(coda)
mc1 <- mcmc(out1)
summary(mc1)

## ----post----------------------------------------------------------------
plot(mc1)

## ------------------------------------------------------------------------
mc1b <- window(mc1, start=101, thin=1)

## ------------------------------------------------------------------------
rejectionRate(mc1b)

## ------------------------------------------------------------------------
library(parallel)
nCores <- 2
cl <- makeCluster(nCores)
clusterExport(cl, c("lm.gibbs", "y", "x"))
clusterSetRNGStream(cl, 3479)
out <- clusterEvalQ(cl, {
    mc <- lm.gibbs(y=y, x=x, niter=1000,
                   start=c(0,0,1), tune=c(0.7,0.7,0.3))
    return(mc)
})
mcp <- as.mcmc.list(lapply(out, function(x) mcmc(x)))

## ----postp---------------------------------------------------------------
plot(mcp)

## ------------------------------------------------------------------------
gelman.diag(mcp) # Point ests. should be <1.1 or so

## ------------------------------------------------------------------------
stopCluster(cl)

## ------------------------------------------------------------------------
jd <- list(y=y, x=x, n=n)
str(jd)

## ------------------------------------------------------------------------
jp <- c("beta0", "beta1", "sigma")

## ------------------------------------------------------------------------
ji <- function() {
    list(beta0=rnorm(1), beta1=rnorm(1), sigmaSq=runif(1))
}
ji()

## ----jm,results='hide'---------------------------------------------------
library(rjags)
jm <- jags.model("lm-JAGS.jag", data=jd, inits=ji, n.chains=3,
                 n.adapt=1000)

## ----jc,results='hide'---------------------------------------------------
jc <- coda.samples(jm, jp, n.iter=5000)

## ------------------------------------------------------------------------
summary(jc)

## ----jc2,results='hide'--------------------------------------------------
jc2 <- coda.samples(jm, jp, n.iter=1000)

## ----jc2-plot------------------------------------------------------------
plot(jc2)

