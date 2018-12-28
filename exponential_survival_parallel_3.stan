/**************************************************************************************/
functions {
	vector lsurv(vector beta, vector theta, real[] x, int [] y) {
        real lp;
	    int NC=y[1];
        int N=y[2];
        int is_censored=y[3];
        real intercept = beta[1];
        vector[NC] betas = beta[2:(1+NC)]; 

        if(is_censored) lp = exponential_lccdf(x[1:N] | exp(intercept+ to_matrix(x[(1+N):(N + N*NC)],N,NC)*betas));
        else lp = exponential_lpdf(x[1:N] | exp(intercept+ to_matrix(x[(1+N):(N+N*NC)],N,NC)*betas));

		return [lp]';
	}
}
/**************************************************************************************/
data {
    int<lower=0> N_uncensored;                                      
    int<lower=0> N_censored;                                        
    int<lower=1> NC;
    int<lower=1> shards;
    matrix[N_censored,NC] X_censored;                               
    matrix[N_uncensored,NC] X_uncensored;                           
    vector<lower=0>[N_censored] times_censored;                          
    vector<lower=0>[N_uncensored] times_uncensored;                       
}
/**************************************************************************************/
transformed data {
    vector[0] theta[shards];
    int<lower = 0> N_censored_ =N_censored / shards;
    int<lower = 0> N_uncensored_ =N_uncensored / shards;
    real x_censored_r[shards, N_censored_ * (1+ NC)];
    int x_censored_i[shards, 3];
    real x_uncensored_r[shards, N_uncensored_ * (1+ NC)];
    int x_uncensored_i[shards, 3];
    {
        int pos = 1;
        for (k in 1:shards) {
            int end= pos + N_censored_-1;
            x_censored_r[k] = to_array_1d(append_col(times_censored[pos:end],X_censored[pos:end]));
            x_censored_i[k,1] = NC;
            x_censored_i[k,2] = N_censored_;
            x_censored_i[k,3] = 1;
            pos += N_censored_;
        }
        pos=1;
        for (k in 1:shards) {
            int end= pos + N_uncensored_-1;
            x_uncensored_r[k] = to_array_1d(append_col(times_uncensored[pos:end],X_uncensored[pos:end]));
            x_uncensored_i[k,1] = NC;
            x_uncensored_i[k,2] = N_uncensored_;
            x_uncensored_i[k,3] = 0;
            pos += N_uncensored_;
        }
    }

}
/**************************************************************************************/
parameters {
    vector[NC] betas;                                     
    real intercept;                                 
}
/**************************************************************************************/
model {
    betas ~ normal(0,2);                                                            
    intercept   ~ normal(-2,2);                                                     
    //target += exponential_lpdf(times_uncensored | exp(intercept+X_uncensored*betas)); 
    target += sum(map_rect(lsurv, append_row(intercept, betas), theta, x_uncensored_r, x_uncensored_i));
    target +=  sum(map_rect(lsurv, append_row(intercept, betas),theta, x_censored_r, x_censored_i)); 
    if(N_censored % shards > 0) 
        target += exponential_lccdf(times_censored[(shards*N_censored_+1):N_censored] | 
                                    exp(intercept + X_censored[(shards*N_censored_+1):N_censored]*betas));
    if(N_uncensored % shards > 0) 
        target += exponential_lpdf(times_uncensored[(shards*N_uncensored_+1):N_uncensored] | 
                                    exp(intercept + X_uncensored[(shards*N_uncensored_+1):N_uncensored]*betas));

}
/**************************************************************************************/
