import numpy as np

def stickbreak(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = np.exp(np.log1p(-v).cumsum(-1))
    # cumprod_one_minus_v = np.cumprod(1-v, axis=-1)
    one_v = np.pad(v, [[0, 0]] * batch_ndims + [[0, 1]], constant_values=1)
    c_one = np.pad(cumprod_one_minus_v, [[0, 0]] * batch_ndims +[[1, 0]], constant_values=1)
    return one_v * c_one

def dp_sb_gmm(y, max_components):
    # Cosntants
    N = y.shape[0]
    K = max_components
    
    # Priors
    alpha = numpyro.sample('alpha', dist.Gamma(1, 1))
    
    with numpyro.plate('mixture_weights', K - 1):
        v = numpyro.sample('v', dist.Beta(1, alpha, K - 1))
    
    eta = stickbreak(v)
    
    with numpyro.plate('components', K):
        sigma_m = numpyro.sample('s_mu', dist.Normal(0., 10)) 
        mu_i = numpyro.sample('mu', dist.Normal(0., 1))
        mu = mu_i*sigma
        sigma = numpyro.sample('sigma', dist.Gamma(1, 1))

    with numpyro.plate('data', N):
        numpyro.sample('obs', NormalMixture(mu[None, :] , sigma[None, :], eta[None, :]),
                       obs=y[:, None])
    
    return