## HMC with TMB used to compute joint density and gradient

library(TMB)


## Linear regression

## set.seed(340)
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
        M <- diag(n.theta) ## Could be based on var(post-samples)
    } 
    Minv <- solve(M)

    ## Use TMB to create log-posterior function and its gradient
    fg <- MakeADFun(data, as.list(theta), DLL="lm_post", silent=TRUE)
    
    ## samples
    samples <- matrix(NA_real_, n.iter, 4)
    colnames(samples) <- c("beta0", "beta1", "logSigma", "H")

    L <- 1
    for(i in 1:n.iter) {
        theta.cand <- theta
        grad.lp.new <- as.vector(fg$gr(theta.cand))
        ## m <- rnorm(n.theta, 0, 1) ## Could use rmvnorm(1, 0, M)
        m <- drop(rmvnorm(1, rep(0, n.theta), M))
        m.new <- m - eps/2 * grad.lp.new
        ## Leapfrog
        for(j in 1:n.leap) {
            ## Full step for position
            theta.cand <- as.vector(theta.cand + eps * Minv %*% m.new)
            grad.lp.new <- as.vector(fg$gr(theta.cand))
            if(j != n.leap) {
                ## Full step for momentum, except at the end
                m.new <- m.new - eps * grad.lp.new
            }
        }
        ## Half-step for momentum
        m.new <- m.new - eps/2 * grad.lp.new
        m.new <- -m.new ## Negate momentum at the end (correct, but not required)
        ## Hamiltonian, defined as -log(post)+log(momentum)
        H.cand <- fg$fn(theta.cand) + 0.5*(m.new %*% Minv %*% m.new)
        H <- fg$fn(theta) + 0.5*(m %*% Minv %*% m)
        ## Metropolis step used to accept proposal with min(1,exp(-Hcand)/exp(-H))
        ## if(runif(1) < exp(-H.cand - -H)) {
        if(runif(1) < exp(H-H.cand)) {
            theta <- theta.cand
            m <- -m.new
            H <- H.cand
        }
        samples[i,] <- c(theta, H)
    }
    return(samples)
}


d <- list(y=y, x=x)

## debugonce(hmc.sample.lm)

samps <- hmc.sample.lm(data=d, n.iter=1000, n.leap=10, eps=c(0.04, 0.04, 0.095))

plot(samps[,1:2])
arrows(samps[-100, 1], samps[-100, 2],
       samps[-1, 1], samps[-1, 2], length=0.1, col="red")

pairs(samps)

library(coda)

mc <- mcmc(samps)

summary(window(mc, start=201))

c(beta, log(sigma))

effectiveSize(window(mc, start=201))

rejectionRate(mc)

plot(mc)

plot(window(mc, start=201))

autocorr.plot(mc)

crosscorr.plot(mc)

cov(mc[,1:3])







## Improve efficiency by using inverse of posterior
## covariance matrix as prior on momentum
## Works great, even with very few leap frog steps


(newM <- solve(cov(mc[,1:3])))

samps2 <- hmc.sample.lm(data=d, n.iter=1000, n.leap=5, eps=c(0.5, 0.5, 0.5), M=newM)

plot(samps2[,1:2])
arrows(samps2[-100, 1], samps2[-100, 2],
       samps2[-1, 1], samps2[-1, 2], length=0.1, col="red")

pairs(samps2)

library(coda)

mc2 <- mcmc(samps2)

effectiveSize(window(mc2, start=201))

rejectionRate(mc2)

plot(mc2)

plot(window(mc2, start=201))

autocorr.plot(mc2)

crosscorr.plot(mc2)



