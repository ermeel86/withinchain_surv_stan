library(rstan)
library(tidyverse)
library(simsurv)
library(broom)
library(bayesplot)
#############################################################################
sm <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/exponential_survival_parallel_2.stan")
sm0 <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/exponential_survival.stan")
#############################################################################
set.seed(42)
N <- 10000
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
#############################################################################
stan_data <- list(N_uncensored=N-N_censored, 
                  N_censored=N_censored, 
                  X_censored=as.matrix(X[msk_censored,]),
                  X_uncensored=as.matrix(X[!msk_censored,]),
                  times_censored=times[msk_censored],
                  times_uncensored = times[!msk_censored],
                  NC=ncol(X),
                  shards=3
)
#############################################################################
Sys.setenv(STAN_NUM_THREADS=1)
start_time <- Sys.time()
fit1 <- sampling(sm, 
                data=stan_data, 
                seed=42, 
                chains=4, 
                cores=1, 
                iter=10000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
Sys.setenv(STAN_NUM_THREADS=2)
start_time <- Sys.time()
fit2 <- sampling(sm, 
                data=stan_data, 
                seed=42, 
                chains=4, 
                cores=1, 
                iter=10000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
Sys.setenv(STAN_NUM_THREADS=3)
start_time <- Sys.time()
fit3 <- sampling(sm, 
                 data=stan_data, 
                 seed=42, 
                 chains=4, 
                 cores=1, 
                 iter=10000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
Sys.setenv(STAN_NUM_THREADS=4)
start_time <- Sys.time()
fit4 <- sampling(sm, 
                 data=stan_data, 
                 seed=42, 
                 chains=4, 
                 cores=1, 
                 iter=10000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
start_time <- Sys.time()
fit0 <- sampling(sm0, 
                data=stan_data, 
                seed=42, 
                chains=4, 
                cores=1, 
                iter=10000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
broom::tidy(fit4, conf.int=T, rhat=T, ess=T)
broom::tidy(fit3, conf.int=T, rhat=T, ess=T)
broom::tidy(fit2, conf.int=T, rhat=T,ess=T)
broom::tidy(fit1, conf.int=T, rhat=T,ess=T)
broom::tidy(fit0, conf.int=T, rhat=T,ess=T)
#############################################################################
df_fit1 <- as.tibble(as.data.frame(fit1))
df_fit4 <- as.tibble(as.data.frame(fit4))
df_fit3 <- as.tibble(as.data.frame(fit3))
df_fit0 <- as.tibble(as.data.frame(fit0))
df_fit2 <- as.tibble(as.data.frame(fit2))
colnames(df_fit0) <- sprintf("%s_0", colnames(df_fit0)) 
colnames(df_fit1) <- sprintf("%s_1", colnames(df_fit1))
colnames(df_fit2) <- sprintf("%s_2", colnames(df_fit2))
colnames(df_fit3) <- sprintf("%s_3", colnames(df_fit3))
colnames(df_fit4) <- sprintf("%s_4", colnames(df_fit4))
df_fits <- bind_cols(df_fit4,df_fit3, df_fit2, df_fit1, df_fit0)
bayesplot::color_scheme_set("red")
bayesplot::mcmc_areas(df_fits, regex_pars = "intercept")
bayesplot::mcmc_areas(df_fits, regex_pars = "betas")
bayesplot::mcmc_intervals(df_fits, regex_pars = "intercept")
bayesplot::mcmc_intervals(df_fits, regex_pars = "betas")
#############################################################################





