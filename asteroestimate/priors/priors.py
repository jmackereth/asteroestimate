import numpy as np
from asteroestimate.detections import probability as prob
from scipy.stats import norm, multivariate_normal

def numax_JHK(J, H, K, parallax, mass=1., AK=None, N_samples=1000):
    """
    Evaluate a prior on numax based on 2MASS magnitudes and Gaia parallax
    INPUT:
        J, H, K, parallax - 2MASS magnitudes and gaia parallax as tuples of [value, 1sigma uncertainty]
        mass - optional mass prior option (not yet implemented, defaults to 1 msun star)
        AK - optional K band extinction
        N_samples -  number of samples from the prior to take and then return (default: 1000)
    OUTPUT:
        out - summary statistics assuming samples are Gaussian distributed (numax_mean, numax_sigma)
        samples - samples from the prior
    HISTORY:
        Written - Mackereth - 08/09/2020 (UoB @ online.tess.science)
    """
    means = np.array([J[0], H[0], K[0], parallax[0]])
    cov = np.zeros((4,4))
    cov[0,0] = J[1]**2
    cov[1,1] = H[1]**2
    cov[2,2] = K[1]**2
    cov[3,3] = parallax[1]**2
    multi_norm = multivariate_normal(means, cov)
    samples = multi_norm.rvs(size=N_samples)
    Jsamp, Hsamp, Ksamp, parallaxsamp = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    numaxsamp = prob.numax_from_JHK(Jsamp, Hsamp, Ksamp, parallaxsamp, mass=mass, AK=AK)
    numax_mean = np.mean(numaxsamp)
    numax_sigma = np.std(numaxsamp)
    return (numax_mean, numax_sigma), numaxsamp
