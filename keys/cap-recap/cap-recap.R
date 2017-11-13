## Simulate non-spatial cap-recap data: M0

N <- 30
K <- 10  # nOccasions
p <- 0.2

## All encounter histories
yall <- matrix(NA, N, K)
for(k in 1:K) {
    yall[,k] <- rbinom(N, 1, p)
}


## Data
y <- yall[rowSums(yall)>0,]
y



## Augment the data
M <- 100
yAug <- matrix(0, M, K)
yAug[1:nrow(y),] <- y
yAug

## JAGS

library(rjags)


jd <- list(yAug=yAug, K=K, M=M)

ji <- function() list(p=runif(1), psi=runif(1),
                      z=rep(1, M))

jp <- c("p", "psi", "N")


jm <- jags.model(file="M0.jag", data=jd, inits=ji)

jp <- coda.samples(jm, jp, n.iter=1000)


plot(jp)

summary(jp)
