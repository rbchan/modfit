
library(TMB)



compile("lm.cpp")
dyn.load(dynlib("lm"))

set.seed(340)
n <- 10000
x <- rnorm(n)
sigma <- 2
eps <- rnorm(n, 0, sigma)
beta <- c(-1, 1)
y <- beta[1] + beta[2]*x + eps

dat <- list(Y=y, x=x)
par <- list(a=0, b=0, logSigma=0)

obj <- MakeADFun(dat, par, DLL="lm", hessian=TRUE)

str(obj)

obj$fn(obj$par)
obj$gr(obj$par)
obj$he(obj$par)

obj$gr(c(0,0,0))
obj$gr(c(1,1,1))
