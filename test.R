library("rstan") # observe startup messages
library(tidyverse)
library(simsurv)
sm <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/exponential_survival_parallel.stan")

N <- 50000
cov <- data.frame(id = 1:N,
                  trt = rbinom(N, 1, 0.5))

# Simulate the event times
dat <- simsurv(lambdas = 0.1, 
               gammas = 1, 
               betas = c(trt = -0.5), 
               x = cov, 
               maxt = 3)
# Merge the simulated event times onto covariate data frame
df <- merge(cov, dat)
N <- nrow(df)
X <- as.matrix(as.integer(pull(df, trt)))
is_censored <- pull(df,status)==0
times <- pull(df,eventtime)
msk_censored <- is_censored == 1
N_censored <- sum(msk_censored)
stan_data <- list(N_uncensored=N-N_censored, 
                  N_censored=N_censored, 
                  X_censored=as.matrix(X[msk_censored,]),
                  X_uncensored=as.matrix(X[!msk_censored,]),
                  times_censored=times[msk_censored],
                  times_uncensored = times[!msk_censored],
                  NC=ncol(X),
                  shards=3
)
Sys.setenv(STAN_NUM_THREADS=2)
start_time <- Sys.time()
fit <- sampling(sm, 
                data=stan_data, 
                seed=42, 
                chains=4, 
                cores=1, 
                iter=10000)
end_time <- Sys.time()
end_time - start_time

