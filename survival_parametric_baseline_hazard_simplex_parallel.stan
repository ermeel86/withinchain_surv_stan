/**************************************************************************************/
functions {
	vector lsurv(vector beta, vector theta, real[] x, int [] y) {
        real lp;
	    int NC=y[1];
        int N=y[2];
        int m=y[3];
        int is_censored=y[4];

        real intercept = beta[1];
        vector[NC] betas = beta[2:(1+NC)]; 
        vector[m] gammas = beta[(2+NC):(1+NC+m)]; // should this not be simplex?
        if(is_censored) lp = sum(-(to_matrix(x[1:N*m],N,m)*gammas) .* exp(intercept+ to_matrix(x[(1+2*N*m):(2*N*m + N*NC)],N,NC)*betas));
        else lp = sum(-(to_matrix(x[1:N*m],N,m)*gammas) .* exp(intercept+ to_matrix(x[(1+2*N*m):(2*N*m + N*NC)],N,NC)*betas) +
                  log(to_matrix(x[N*m+1:2*N*m], N, m)*gammas) + to_matrix(x[(1+2*N*m):(2*N*m + N*NC)],N,NC)*betas + intercept);
        
		return [lp]';
	}
}
/**************************************************************************************/
data {
    int<lower=0> N_uncensored;                                   
    int<lower=0> N_censored;                                        
    int<lower=1> m;                                                 
    int<lower=0> NC;                                                
    matrix[N_censored,NC] X_censored;                               
    matrix[N_uncensored,NC] X_uncensored;                                                
    matrix[N_uncensored,m] m_spline_basis_evals_uncensored;                  
    matrix[N_uncensored,m] i_spline_basis_evals_uncensored;   
    matrix[N_censored,m] i_spline_basis_evals_censored;
    real<lower=0> alpha;
    int<lower=1> shards;
}
/**************************************************************************************/
transformed data {
    vector[0] theta[shards];
    int<lower = 0> N_censored_ =N_censored / shards;
    int<lower = 0> N_uncensored_ =N_uncensored / shards;
    real x_censored_r[shards, N_censored_ * (1+ NC + 2*m)];
    int x_censored_i[shards, 4];
    real x_uncensored_r[shards, N_uncensored_ * (1+ NC + 2*m)];
    int x_uncensored_i[shards, 4];
    vector[m] alphas=rep_vector(alpha, m);

    {
        int pos = 1;
        for (k in 1:shards) {
            int end= pos + N_censored_-1;
            x_censored_r[k] = to_array_1d(
                                          append_col(i_spline_basis_evals_censored[pos:end],
                                                     append_col(
                                                                to_matrix(rep_vector(0, m*N_censored_),N_censored_,m), 
                                                                X_censored[pos:end])
                                                                )
                                          );
            x_censored_i[k,1] = NC;
            x_censored_i[k,2] = N_censored_;
            x_censored_i[k,3] = m;
            x_censored_i[k,4] = 1;
            pos += N_censored_;
        }
        pos=1;
        for (k in 1:shards) {
            int end= pos + N_uncensored_-1;
            x_uncensored_r[k] = to_array_1d(
                                            append_col(i_spline_basis_evals_uncensored[pos:end],
                                                       append_col(
                                                                  m_spline_basis_evals_uncensored[pos:end], 
                                                                  X_uncensored[pos:end])
                                                                 )
                                            );
            x_uncensored_i[k,1] = NC;
            x_uncensored_i[k,2] = N_uncensored_;
            x_uncensored_i[k,3] = m;
            x_uncensored_i[k,4] = 0;
            pos += N_uncensored_;
        }
    }

}
/**************************************************************************************/
parameters {
    simplex[m] gammas;          
    vector[NC] betas;                                            
    real intercept;   
}
/**************************************************************************************/
model {
    betas ~ normal(0,2);
    intercept   ~ normal(0,2);
    gammas ~ dirichlet(alphas);

    target += sum(map_rect(lsurv, append_row(intercept, append_row(betas,gammas)), theta, x_uncensored_r, x_uncensored_i));
    target +=  sum(map_rect(lsurv, append_row(intercept, append_row(betas,gammas)),theta, x_censored_r, x_censored_i)); 
    if(N_censored % shards > 0)
        target += -(i_spline_basis_evals_censored[(shards*N_censored_+1):N_censored]*gammas) .* exp(X_censored[(shards*N_censored_+1):N_censored]*betas + intercept);
    if(N_uncensored % shards > 0) { 
        target += -(i_spline_basis_evals_uncensored[(shards*N_uncensored_+1):N_uncensored]*gammas) .* exp(X_uncensored[(shards*N_uncensored_+1):N_uncensored]*betas + intercept);
        target +=  log(m_spline_basis_evals_uncensored[(shards*N_uncensored_+1):N_uncensored]*gammas) + X_uncensored[(shards*N_uncensored_+1):N_uncensored]*betas + intercept;
    }
}
/**************************************************************************************/
