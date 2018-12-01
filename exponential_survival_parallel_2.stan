/**************************************************************************************/
functions {
	vector lsurv(vector beta, vector theta, real[] x, int [] y) {
        real lp;
	    int NC=y[1];
        int N=y[2];
        real intercept = beta[1];
        vector[NC] betas = beta[2:(1+NC)]; 
        lp = exponential_lccdf(x[1:N] | exp(intercept+ to_matrix(x[(1+N):(N + N*NC)],N,NC)*betas));
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
    int<lower = 0> J =N_censored / shards;
    real x_r[shards, J * (1+ NC)];
    int x_i[shards, 2];
    {
        int pos = 1;
        for (k in 1:shards) {
            int end= pos + J-1;
            x_r[k] = to_array_1d(append_col(times_censored[pos:end],X_censored[pos:end]));
            x_i[k,1] = NC;
            x_i[k,2] = J;
            pos += J;
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
    target += exponential_lpdf(times_uncensored | exp(intercept+X_uncensored*betas)); 
    target +=  sum(map_rect(lsurv, append_row(intercept, betas),theta, x_r, x_i)); 
    if(N_censored % shards > 0) 
        target += exponential_lccdf(times_censored[(shards*J+1):N_censored] | 
                                    exp(intercept + X_censored[(shards*J+1):N_censored]*betas));

}
/**************************************************************************************/
