library(rstan)
options(mc.cores = 1)
rstan_options(auto_write = TRUE)
library(tidyverse)
library(simsurv)
library(broom)
library(bayesplot)
library(cowplot)
library(splines2)
#############################################################################
sm <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/exponential_survival_parallel_3.stan")
sm0 <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/exponential_survival.stan")
smm <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/survival_parametric_baseline_hazard_simplex_parallel.stan")
smm0 <- stan_model("~/Desktop/Stan/Within_Chain_Parallelisation/survival_parametric_baseline_hazard_simplex.stan")
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
                  shards=5
)
#############################################################################
time_range <- range(times)
time_min <- time_range[1]
time_max <- time_range[2]

knots <- quantile(times[!msk_censored], probs = c(.05, .35, .65, .95))
dirichlet_alpha<- 2
nknots <- length(knots)
mspline_degree<-3

i_spline_basis_evals <- iSpline(times, knots=knots, degree=mspline_degree,
                                intercept=FALSE, Boundary.knots = c(0, time_max))
m_spline_basis_evals <- deriv(i_spline_basis_evals)
i_spline_basis_evals_censored <- i_spline_basis_evals[msk_censored,]
i_spline_basis_evals_uncensored <- i_spline_basis_evals[!msk_censored,]
m_spline_basis_evals_uncensored <- m_spline_basis_evals[!msk_censored,]
nbasis <- dim(i_spline_basis_evals_censored)[2]




stan_data_m <- list(N_uncensored=N-N_censored, 
                  N_censored=N_censored, 
                  X_censored=as.matrix(X[msk_censored,]),
                  X_uncensored=as.matrix(X[!msk_censored,]),
                  times_censored=times[msk_censored],
                  times_uncensored = times[!msk_censored],
                  NC=ncol(X),
                  shards=5,
                  m_spline_basis_evals_uncensored=m_spline_basis_evals_uncensored, 
                  i_spline_basis_evals_uncensored=i_spline_basis_evals_uncensored,
                  i_spline_basis_evals_censored=i_spline_basis_evals_censored,
                  alpha=dirichlet_alpha,
                  m=nbasis
)
#############################################################################
# MSpline sequential
start_time <- Sys.time()
fit_m0 <- sampling(smm0, 
                 data=stan_data_m, 
                 seed=42, 
                 init=0,
                 chains=1, 
                 cores=1, 
                 iter=20000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
Sys.setenv(STAN_NUM_THREADS=3)
stan_data_m[["shards"]] <- 5
start_time <- Sys.time()
fit_mm <- sampling(smm, 
                  data=stan_data_m, 
                  seed=42, 
                  init=0,
                  chains=1, 
                  cores=1, 
                  iter=20000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
Sys.setenv(STAN_NUM_THREADS=3)
stan_data_m[["shards"]] <- 6
start_time <- Sys.time()
fit_mm <- sampling(smm, 
                   data=stan_data_m, 
                   seed=42, 
                   init=0,
                   chains=1, 
                   cores=1, 
                   iter=20000)
end_time <- Sys.time()
end_time - start_time
#############################################################################
df_m0 <- as.tibble(as.data.frame(fit_m0))
df_m <- as.tibble(as.data.frame(fit_mm))
colnames(df_m) <- sprintf("%s_parallel",colnames(df_m))
df_comp <- bind_cols(df_m0, df_m)
bayesplot::mcmc_areas_ridges(df_comp, regex_pars = "betas")
bayesplot::mcmc_areas_ridges(df_comp, regex_pars = "gammas")
bayesplot::mcmc_intervals(df_comp,regex_pars="gammas")
#############################################################################
#############################################################################
# Constant baseline parallel
Sys.setenv(STAN_NUM_THREADS=3)
start_time <- Sys.time()
fit2 <- sampling(sm, 
                data=stan_data, 
                seed=42, 
                init=0,
                chains=1, 
                cores=1, 
                iter=20000)
end_time <- Sys.time()
end_time - start_time
#############################################################################