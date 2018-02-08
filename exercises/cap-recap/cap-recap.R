## Simulate non-spatial cap-recap data: M0

N <- 30
K <- 10  # nOccasions
p <- 0.5

1 - (1-p)^K

## All encounter histories
yall <- matrix(NA, N, K)
for(k in 1:K) {
    yall[,k] <- rbinom(N, 1, p)
}


## Data
y <- yall[rowSums(yall)>0,]
y


## Summary stats
rowSums(y)
table(rowSums(y))
colSums(y)


## Augment the data
M <- 100
yAug <- matrix(0, M, K)
yAug[1:nrow(y),] <- y
yAug






## JAGS

library(rjags)

## Model M0
jd <- list(yAug=yAug, K=K, M=M)

ji <- function() list(p=runif(1), psi=runif(1),
                      z=rep(1, M))

jp <- c("p", "psi", "N", "deviance")

load.module("dic")

jm <- jags.model(file="M0.jag", data=jd, inits=ji)

jp <- coda.samples(jm, jp, n.iter=1000)


plot(jp)

summary(jp)






## Model Mt
jd <- list(yAug=yAug, K=K, M=M)

ji.Mt <- function() list(p=runif(jd$K), psi=runif(1),
                      z=rep(1, M))
ji.Mt()

jp <- c("p", "psi", "N", "deviance")


jm.Mt <- jags.model(file="Mt.jag", data=jd, inits=ji.Mt)

jp.Mt <- coda.samples(jm.Mt, jp, n.iter=1000)


plot(jp.Mt, ask=TRUE)

(sp.Mt <- summary(jp.Mt))

plot(1:10, sp.Mt$quantile[2:11,3], ylim=c(0,1))
segments(1:10, sp.Mt$quantile[2:11,1], 1:10, sp.Mt$quantile[2:11,5])









## Model Mb

firstcap <- apply(y, 1, function(x) min(which(x==1)))
firstcap

prevcap <- matrix(1, M, ncol(y))
for(i in 1:nrow(y)) {
    prevcap[i, (firstcap[i]+1):ncol(y)] <- 2
}

prevcap

jd.Mb <- list(yAug=yAug, K=K, M=M, prevCap=prevcap)

ji.Mb <- function() list(p=runif(2), psi=runif(1),
                         z=rep(1, M))
ji.Mb()

jp <- c("p", "psi", "N", "deviance")


jm.Mb <- jags.model(file="Mb.jag", data=jd.Mb, inits=ji.Mb)

jp.Mb <- coda.samples(jm.Mb, jp, n.iter=1000)


plot(jp.Mb, ask=TRUE)

(sp.Mb <- summary(jp.Mb))

plot(1:2, sp.Mb$quantile[3:4,3], ylim=c(0,1), xlim=c(0,3))
segments(1:2, sp.Mb$quantile[3:4,1], 1:2, sp.Mb$quantile[3:4,5])
