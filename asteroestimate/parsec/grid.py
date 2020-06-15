import numpy as np
from astropy.io import fits
import os
from scipy.stats import multivariate_normal, norm
from scipy.interpolate import interp1d
datapath = os.environ['ASTEROESTIMATE_DATA']

def fullPARSECgrid():
    """ load the pre-compiled grid with Chabrier (2001) IMF and global seismic params from scaling relations """
    path = os.path.join(datapath,'full_parsec_grid.fits')
    if os.path.exists(path):
        file = fits.getdata(path)
        return file
    else:
        raise IOError('PARSEC grid file is not present, either download or compile the grid using grid.compileParsec().')

def p_jhk(fullgrid, j,h,k,j_err,h_err,k_err, mask=None):
    "get the probability of each isochrone point for a given data point"
    if mask is None:
        mask = np.ones(len(fullgrid['J']), dtype=bool)
    mean = np.array([j,h,k])
    cov = np.array([[j_err**2,0,0],[0,h_err**2,0],[0,0,k_err**2]])
    rv = multivariate_normal(mean, cov)
    jhk_grid = np.dstack([fullgrid['J'][mask], fullgrid['H'][mask], fullgrid['K'][mask]])[0]
    return rv.pdf(jhk_grid)

def p_jminuskh(fullgrid, jk,h,jk_err,h_err):
    "get the probability of each isochrone point for a given data point"
    mean = np.array([jk,h])
    cov = np.array([[jk_err**2,0],[0,h_err**2]])
    rv = multivariate_normal(mean, cov)
    jhk_grid = np.dstack([fullgrid['J']-fullgrid['K'], fullgrid['H']])[0]
    return rv.pdf(jhk_grid)

def sample_from_grid_jhk(fullgrid, j,h,k,j_err,h_err,k_err, mask=None, N=100):
    "sample from the isochrone grid around a point in 2MASS photometry"
    probs = p_jhk(fullgrid, j,h,k,j_err,h_err,k_err, mask=mask)
    if mask is None:
        mask = np.ones(len(fullgrid['J']), dtype=bool)
    weights = probs*(fullgrid['delta_M'][mask]*(fullgrid['age'][mask]/fullgrid['Z'][mask]))
    sort = np.argsort(weights)
    tinter = interp1d(np.cumsum(weights[sort])/np.sum(weights), range(len(weights[sort])), kind='linear')
    randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
    if np.any(randinds < 0):
        #if the star is outside the grid - just sample completely random points from all the isochrones?
        randinds = np.random.choice(len(fullgrid[sort]), size=N)
    return fullgrid[mask][sort][randinds]

def sample_from_grid_jminuskh(fullgrid, jk, h, jk_err,h_err, N=100):
    "sample from the isochrone grid around a point in 2MASS photometry"
    probs = p_jminuskh(fullgrid, jk,h,jk_err,h_err)
    weights = probs*(fullgrid['delta_M']*(fullgrid['age']/fullgrid['Z']))
    sort = np.argsort(weights)
    tinter = interp1d(np.cumsum(weights[sort])/np.sum(weights), range(len(weights[sort])), kind='linear')
    randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
    if np.any(randinds < 0):
        #if the star is outside the grid - just sample completely random points from all the isochrones?
        randinds = np.random.choice(len(fullgrid[sort]), size=N)
    return fullgrid[sort][randinds]
