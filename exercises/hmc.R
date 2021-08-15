## HMC with TMB used to compute joint density and gradient

library(TMB)


## Linear regression

set.seed(340)
n <- 100
x <- rnorm(n)
sigma <- 2
eps <- rnorm(n, 0, sigma)
beta <- c(-1, 1)
y <- beta[1] + beta[2]*x + eps

plot(x,y)


## TBM model for lin reg with normal priors

compile("lm_post.cpp")
dyn.load(dynlib("lm_post"))

d <- list(x=x, y=y)
par <- list(beta0=0, beta1=1, logSigma=0)
fg <- MakeADFun(d, par, DLL="lm_post")



hmc.sample.lm <- function(data, n.iter, n.leap, eps) {

    ## Initial values
    beta0 <- 2
    beta1 <- 0
    sigma <- 5
    theta <- c(beta0=beta0, beta1=beta1, logSigma=log(sigma))
    n.theta <- length(theta)
    M <- diag(n.theta)
    Minv <- M

    ## Use TMB to create log-posterior function and its gradient
    fg <- MakeADFun(data, as.list(theta), DLL="lm_post", silent=TRUE)
    
    ## samples
    samples <- matrix(NA_real_, n.iter, 3)
    colnames(samples) <- c("beta0", "beta1", "logSigma")

    L <- 1
    for(i in 1:n.iter) {
        m <- rnorm(n.theta, 0, 1) ## Could use rmvnorm(1, 0, M)
        m.new <- m
        ## Leapfrog
        if(L>n.leap)
            L <- 2
        for(j in 1:L) {
            ## for(j in 1:n.leap) {
            grad.lp <- as.vector(fg$gr(theta))
            m.new <- m + eps/2 * grad.lp
            theta.cand <- theta + eps * Minv %*% m.new
            grad.lp.new <- as.vector(fg$gr(theta.cand))
            if(j == n.leap) {
                m.new <- m.new + eps/2 * grad.lp.new
            }
        }
        L <- L+1 ## Cycle through length of leapfrog trajectories
        num <- fg$fn(theta.cand) - 0.5*(m.new %*% Minv %*% m.new)
        den <- fg$fn(theta) - 0.5*(m %*% Minv %*% m)
        if(runif(1) < exp(num - den)) {
            theta <- theta.cand
            m <- -m.new
        }
        samples[i,] <- theta
    }
    return(samples)
}


d <- list(y=y, x=x)

## debugonce(hmc.sample.lm)

samps <- hmc.sample.lm(data=d, n.iter=1000, n.leap=4*2, eps=c(0.2, 0.2, 0.1))

plot(samps[,1:2])
arrows(samps[-100, 1], samps[-100, 2],
       samps[-1, 1], samps[-1, 2], length=0.1, col="red")

pairs(samps)

library(coda)

mc <- mcmc(samps)

effectiveSize(window(mc, start=201))

rejectionRate(mc)

plot(mc)

plot(window(mc, start=101))
