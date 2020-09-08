import numpy as np
import detections.probability as prob

def numax_JHK(J, H, K, parallax, mass=None, AK=None, N_samples=1000):
    """
    Evaluate a prior on numax based on 2MASS magnitudes and Gaia parallax
    INPUT:
        J, H, K, parallax - 2MASS magnitudes and gaia parallax as tuples of [value, 1sigma uncertainty]
        mass - optional mass prior option (not yet implemented)
        AK - optional K band extinction
        N_samples -  number of samples from the prior to return (default: 1000)
    OUTPUT:
        out - summary statistics assuming samples are Gaussian distributed (numax_mean, numax_sigma)
        samples - samples from the prior
    HISTORY:
        Written - Mackereth - 08/09/2020 (UoB @ online.tess.science)
    """
    return (numax_mean, numax_sigma), samples
