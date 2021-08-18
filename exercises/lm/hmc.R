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



hmc.sample.lm <- function(data, n.iter, n.leap, eps, M=NULL) {

    require(mvtnorm)
    
    ## Initial values
    beta0 <- 2
    beta1 <- 0
    sigma <- 5
    theta <- c(beta0=beta0, beta1=beta1, logSigma=log(sigma))
    n.theta <- length(theta)
    if(is.null(M)) {
        M <- diag(n.theta)
        Minv <- M
    } else {
        if(!isTRUE(all.equal(dim(M), c(n.theta, n.theta))))
            stop("M should be a correlation matrix with dimensions", n.theta, "by", n.theta)
        Minv <- solve(M)
    }
    zeros.m <- rep(0, n.theta)

    ## Use TMB to create log-posterior function and its gradient
    fg <- MakeADFun(data, as.list(theta), DLL="lm_post", silent=TRUE)
    
    ## samples
    samples <- matrix(NA_real_, n.iter, 3)
    colnames(samples) <- c("beta0", "beta1", "logSigma")

    for(i in 1:n.iter) {
        theta.cand <- theta
        ## m <- rnorm(n.theta, 0, 1) 
        m <- drop(rmvnorm(1, zeros.m, M))
        m.new <- m
        ## Leapfrog
        ## if(i==1)
        ##     plot(theta[1], theta[2], xlim=c(-5, 3), ylim=c(-3, 4))
        for(j in 1:n.leap) {
            ## theta.old <- theta.cand
            grad.lp <- as.vector(fg$gr(theta.cand))
            m.new <- m.new + eps/2 * grad.lp
            theta.cand <- theta.cand + eps * Minv %*% m.new
            grad.lp.new <- as.vector(fg$gr(theta.cand))
            if(j == n.leap) {
                m.new <- m.new + eps/2 * grad.lp.new
            }
            ## if(i==1) {
            ##     points(theta.cand[1], theta.cand[2])
            ##     arrows(theta.old[1], theta.old[2], theta.cand[1], theta.cand[2],
            ##            length=0.1, col="red")
            ## }
        }
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

samps <- hmc.sample.lm(data=d, n.iter=1000, n.leap=4*1, eps=c(0.05, 0.05, 0.03))

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



samps2 <- hmc.sample.lm(data=d, n.iter=1000, n.leap=80, eps=sqrt(diag(cov(samps)))/2,
                        M=diag(3)) ##cor(samps))

mc2 <- mcmc(samps2)

effectiveSize(window(mc2, start=201))

rejectionRate(mc2)
