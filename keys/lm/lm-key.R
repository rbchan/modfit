
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



## Minimize neg log-like

fm <- optim(starts, nll, hessian=TRUE)
fm

mles <- fm$par

vcov <- solve(fm$hessian)

SEs <- sqrt(diag(vcov))

cbind(Est=mles, SE=SEs)
