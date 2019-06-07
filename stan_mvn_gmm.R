
# see https://maggielieu.com/2017/03/21/multivariate-gaussian-mixture-model-done-properly/
#  by Maggie Lieu -- demo of multivariate GMM in Stan 

# status: I am running into label switching with multiple chains, even at D=2 and K=4
# running longer chains does not help. seems to be less so the case if running a single chain

library(MASS)
library(rstan)

options(mc.cores = parallel::detectCores())

rstan_options(auto_write = TRUE)


#first cluster
mu1=c(0,0,0,0)
sigma1=matrix(c(0.1,0,0,0,0,0.1,0,0,0,0,0.1,0,0,0,0,0.1),ncol=4,nrow=4, byrow=TRUE)
norm1=mvrnorm(50, mu1, sigma1)

#second cluster
mu2=c(7,7,7,7)
sigma2=sigma1
norm2=mvrnorm(50, mu2, sigma2)

#third cluster
mu3=c(3,3,3,3)
sigma3=sigma1
norm3=mvrnorm(50, mu3, sigma3)

norms=rbind(norm1,norm2,norm3) #combine the 3 mixtures together
N=150 #total number of data points 
Dim=4 #number of dimensions
y=array(as.vector(norms), dim=c(N,Dim))
mixture_data=list(N=N, D=4, K=3, y=y)


mixture_model<-'
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 vector[D] y[N]; //data
}

parameters {
 simplex[K] theta; //mixing proportions
 ordered[D] mu[K]; //mixture component means
 cholesky_factor_corr[D] L[K]; //cholesky factor of covariance
}

model {
 real ps[K];
 
 for(k in 1:K){
 mu[k] ~ normal(0,3);
 L[k] ~ lkj_corr_cholesky(4);
 }
 

 for (n in 1:N){
 for (k in 1:K){
 ps[k] = log(theta[k])+multi_normal_cholesky_lpdf(y[n] | mu[k], L[k]); //increment log probability of the gaussian
 }
 target += log_sum_exp(ps);
 }

}

'

fit=stan(model_code=mixture_model, data=mixture_data, iter=55000, warmup=5000, chains=8)

print(fit)

