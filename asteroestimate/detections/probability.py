import numpy as np
from scipy.stats import chi2, multivariate_normal, norm
from scipy.interpolate import interp1d
import asteroestimate.detections.noise as noise
import asteroestimate.bolometric.polynomial as polybcs
import asteroestimate.parsec.grid as grid
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

numax_sun = 3150 # uHz
dnu_sun = 135.1 # uHz
teff_sun = 5777 # K
taugran_sun = 210 # s
teffred_sun = 8907 # K

obs_available = ['kepler-sc', 'kepler-lc', 'tess-ffi']


def from_phot(G, BP, RP, J, H, K, parallax, s=1., deltaT=1550., Amax_sun=2.5, D=1., obs='kepler-sc', T=30., pfalse=0.01, mass=1., AK=None, numax_limit=None, return_SNR=False):
    """
    Seismic detection probability from Gaia and 2MASS photometry (and Parallax)
    INPUT:
        G, BP, RP, J, H, K - Gaia and 2MASS photometry
        parallax - parallax from Gaia/other in mas
        s, deltaT, Amax_sun - parameters for estimating the stellar oscillation signal (see Chaplin et al. 2011)
        obs - the seismic observation mode, can be kepler-sc, kepler-lc, tess-ffi or tess-ctl currently
        T - the length of the observations in days (e.g. 27 for a single sector of TESS)
        pfalse - the probability of a false positive
        mass - an estimate of the stellar mass, can either be a constant (float) for the whole sample, samples for each star based on some prior (N,N_samples), or use 'giants'/'dwarfs' for a prior for these populations
        AK - K band extinction
        numax_limit - lower limt on detectable nu max (optional)
        return_SNR - return the expected seismic SNR of the observation
    OUTPUT:
        probs - the detecton probability (1. for near definite detection)
        SNR - if return_SNR, the predicted signal-to-noise ratio of the observation
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    if obs not in obs_available:
        raise IOError('%s is not currently implemented as an observation mode, check documentation or probability.obs_available for available modes.' % obskm)
    if AK is not None:
        JK = J-K-1.5*AK
    else:
        JK = J-K
    tlum = Kmag_to_lum(K, JK, parallax, AK=AK, Mbol_sun=4.67) #luminosity in Lsun
    tteff = J_K_Teff(JK) #teff in K
    trad = np.sqrt(tlum/(tteff/teff_sun)**4)
    if isinstance(mass, (int, float,np.float32,np.float64)):
        tmass = mass
        tnumax = numax(tmass, tteff, trad)
        if AK is not None:
            snrtot = SNR_tot(G, BP, RP, J-2.5*AK, H-1.55*AK, K-AK, tlum, tmass, tteff, trad, tnumax, s=s, deltaT=deltaT, Amax_sun=Amax_sun, D=D, obs=obs)
        else:
            snrtot = SNR_tot(G, BP, RP, J, H, K, tlum, tmass, tteff, trad, tnumax, s=s, deltaT=deltaT, Amax_sun=Amax_sun, D=D, obs=obs)
        probs = prob(snrtot, tnumax, T, pfalse)
        if numax_limit is not None:
            probs[tnumax < numax_limit] = 0.
        if return_SNR:
            return probs, snrtot
        return probs
    if isinstance(mass, np.ndarray):
        tmass = mass
        tnumax = numax(tmass, tteff, trad)
        if AK is not None:
            snrtot = SNR_tot(G, BP, RP, J-2.5*AK, H-1.55*AK, K-AK, tlum, tmass, tteff, trad, tnumax, s=s, deltaT=deltaT, Amax_sun=Amax_sun, D=D, obs=obs)
        else:
            snrtot = SNR_tot(G, BP, RP, J, H, K, tlum, tmass, tteff, trad, tnumax, s=s, deltaT=deltaT, Amax_sun=Amax_sun, D=D, obs=obs)
        probs = prob(snrtot, tnumax, T, pfalse)
        if numax_limit is not None:
            probs[tnumax < numax_limit] = 0.
        if return_SNR:
            return probs, snrtot
        return probs
    elif mass == 'giants':
        if isinstance(T, (float,np.float32,np.float64)):
            T = np.ones(len(G))*T
        ndata = len(J)
        msamples = np.random.lognormal(mean=np.log(1.2), sigma=0.4, size=ndata*100)
        tnumax = numax(msamples, np.repeat(tteff,100), np.repeat(trad,100))
        snrtots = SNR_tot(np.repeat(G,100),np.repeat(BP,100),np.repeat(RP,100),np.repeat(J,100),np.repeat(H,100),np.repeat(K,100),
                          np.repeat(tlum,100), msamples, np.repeat(tteff,100), np.repeat(trad,100), tnumax,
                          s=s, deltaT=deltaT, Amax_sun=Amax_sun, obs=obs)
        probs = prob(snrtots,tnumax,np.repeat(T,100),pfalse)
        probs = probs.reshape(ndata,100)
        probs = np.median(probs, axis=1)
        if numax_limit is not None:
            probs[np.median(tnumax.reshape(ndata,100), axis=1) < numax_limit] = 0.
        if return_SNR:
            return probs, np.median(snrtots.reshape(ndata,100),axis=1)
        return probs

def do_one_grid(i, G, BP, RP, J, H, K, J_err, H_err, K_err, j, h, k, j_err, h_err, k_err, N, fullgrid,s,deltaT,Amax_sun, obs, T, pfalse):
    tG = np.repeat(G[i], N)
    tBP = np.repeat(BP[i], N)
    tRP = np.repeat(RP[i], N)
    tJ = norm(J[i], J_err[i]).rvs(N)
    tH = norm(H[i], H_err[i]).rvs(N)
    tK = norm(K[i], K_err[i]).rvs(N)
    samples = grid.sample_from_grid(fullgrid,j[i],h[i],k[i],j_err[i],h_err[i],k_err[i], mask=None, N=N, p='mags')
    snrtot = SNR_tot(tG, tBP, tRP, tJ, tH, tK, samples['luminosity'], samples['M_act'], samples['teff'], samples['radius'], samples['numax'], s=s, deltaT=deltaT, Amax_sun=Amax_sun, obs=obs)
    tprobs = prob(snrtot,samples['numax'],np.repeat(T[i],N),pfalse)
    return np.nanmedian(tprobs)


def from_grid(G, BP, RP, J, H, K, parallax, J_err, H_err, K_err, parallax_err, s=1., deltaT=1550., Amax_sun=2.5, obs='kepler-sc', T=30., pfalse=0.01, AK=None, return_samples=False,  N=100, multiprocess=None, ptype='colormag'):
    """
    Seismic detection probability from Gaia and 2MASS photometry (and Parallax) using the PARSEC isochrone grid (quite slow for large samples!)
    INPUT:
        G, BP, RP, J, H, K - Gaia and 2MASS photometry
        parallax - parallax from Gaia/other in mas
        s, deltaT, Amax_sun - parameters for estimating the stellar oscillation signal (see Chaplin et al. 2011)
        obs - the seismic observation mode, can be kepler-sc, kepler-lc, tess-ffi or tess-ctl currently
        T - the length of the observations in days (e.g. 27 for a single sector of TESS)
        pfalse - the probability of a false positive
        AK - K band extinction
        numax_limit - lower limt on detectable nu max (optional)
        return_SNR - return the expected seismic SNR of the observation
    OUTPUT:
        probs - the detecton probability (1. for near definite detection)
        SNR - if return_SNR, the predicted signal-to-noise ratio of the observation
    HISTORY:
        11/06/2020 - written - J T Mackereth (UoB)
    """
    if AK is None:
        AK = np.zeros(len(G))
    fullgrid = grid.fullPARSECgrid()
    distmod = 5*np.log10(1000/parallax)-5
    error_distmod = (-5/(parallax*np.log(10)))*parallax_err
    j,h,k = J-distmod-(2.5*AK), H-distmod-(1.55*AK), K-distmod-AK
    j_err, h_err, k_err = np.sqrt(J_err**2+error_distmod**2), np.sqrt(H_err**2+error_distmod**2), np.sqrt(K_err**2+error_distmod**2)
    jk = J-K-1.5*AK
    jk_err = np.sqrt(J_err**2+K_err**2)
    if multiprocess is None:
        if return_samples:
            probs = np.zeros((len(G),N))
            allsamples = np.zeros((len(G), N, 12))
        else:
            probs = np.zeros(len(G))
        for i in tqdm.tqdm(range(len(G))):
            tG = np.repeat(G[i], N)
            tBP = np.repeat(BP[i], N)
            tRP = np.repeat(RP[i], N)
            tJ = norm(J[i], J_err[i]).rvs(N)
            tH = norm(H[i], H_err[i]).rvs(N)
            tK = norm(K[i], K_err[i]).rvs(N)
            if ptype == 'mags':
                samples = grid.sample_from_grid_jhk(fullgrid,j[i],h[i],k[i],j_err[i],h_err[i],k_err[i], mask=None, N=N)
            if ptype == 'colormag':
                samples = grid.sample_from_grid_jminuskh(fullgrid,jk[i],h[i], jk_err[i],h_err[i], N=N)
            snrtot = SNR_tot(tG, tBP, tRP, tJ, tH, tK, samples['luminosity'], samples['M_act'], samples['teff'], samples['radius'], samples['numax'], s=s, deltaT=deltaT, Amax_sun=Amax_sun, obs=obs)
            tprobs = prob(snrtot,samples['numax'],np.repeat(T[i],N),pfalse)
            if return_samples:
                probs[i] = tprobs
                allsamples[i] = np.dstack([samples['teff'], samples['radius'], samples['logg'], samples['luminosity'], samples['M_act'], samples['Z'], samples['age'], samples['numax'], samples['dnu'], samples['J'], samples['H'], samples['K']])[0]
            else:
                probs[i] = np.nanmedian(tprobs)
    else:
        do_one = partial(do_one_grid, G=G, BP=BP, RP=RP, J=J, H=H, K=K, J_err=J_err, H_err=H_err, K_err=K_err, j=j, h=h, k=k, j_err=j_err, h_err=h_err, k_err=k_err, N=N, fullgrid=fullgrid, s=s, deltaT=deltaT,Amax_sun=Amax_sun, obs=obs, T=T, pfalse=pfalse)
        with Pool(processes=multiprocess) as pool:
            probs = np.array(list(tqdm.tqdm_notebook(pool.imap(do_one, range(len(G))), total=len(G))))
    if return_samples:
        return probs, allsamples
    return probs



def numax_from_JHK(J, H, K, parallax, mass=1., return_samples=False, AK=None):
    """
    predict frequency at maximum power from 2MASS photometry and Gaia parallax
    INPUT:
        J, H, K - 2MASS photometry
        parallax - parallax from Gaia/other in mas
        mass - an estimate of the stellar mass, can either be a constant (float) for the whole sample, samples for each star based on some prior (N,N_samples), or use 'giants'/'dwarfs' for a prior for these populations
        return_samples - return the samples of numax based on the input mass samples
        return_lum - return the luminosity based on JHK photometry
        AK - the K band extinction
    OUTPUT:
        numax - the predicted numax in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tlum = Kmag_to_lum(K, J-K, parallax, AK=AK, Mbol_sun=4.67) #luminosity in Lsun
    if AK is not None:
        tteff = J_K_Teff(J-K-1.5*AK) #teff in K
    else:
        tteff = J_K_Teff(J-K)
    tteff /= teff_sun
    trad = np.sqrt(tlum/tteff**4)
    if isinstance(mass, (int, float,np.float32,np.float64,np.ndarray)):
        tmass = mass
        tnumax = numax(tmass, tteff*teff_sun, trad)
        return tnumax
    elif mass == 'giants':
        ndata = len(J)
        msamples = np.random.lognormal(mean=np.log(1.2), sigma=0.4, size=ndata*100)#sample_kroupa(ndata*100)
        tnumax = numax(msamples, np.repeat(tteff,100)*teff_sun, np.repeat(trad,100))
        tnumax =  tnumax.reshape(ndata,100)
        if return_samples:
            return tnumax
        return np.median(tnumax, axis=1)


def numax_from_luminosity_teff(luminosity, teff, mass=1., return_samples=False, AK=None):
    """
    predict frequency at maximum power from 2MASS photometry and Gaia parallax
    INPUT:
        luminosity - luminosity in L_sun
        teff - effective temperature in K
        mass - an estimate of the stellar mass, can either be a constant (float) for the whole sample, samples for each star based on some prior (N,N_samples), or use 'giants'/'dwarfs' for a prior for these populations
        return_samples - return the samples of numax based on the input mass samples
        return_lum - return the luminosity based on JHK photometry
        AK - the K band extinction
    OUTPUT:
        numax - the predicted numax in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tlum = luminosity #teff in K
    tteff = teff/teff_sun
    trad = np.sqrt(tlum/tteff**4)
    if isinstance(mass, (int, float,np.float32,np.float64)):
        tmass = mass
        tnumax = numax(tmass, tteff*teff_sun, trad)
        return tnumax
    elif mass == 'giants':
        ndata = len(J)
        msamples = np.random.lognormal(mean=np.log(1.2), sigma=0.4, size=ndata*100)#sample_kroupa(ndata*100)
        tnumax = numax(msamples, np.repeat(tteff,100)*teff_sun, np.repeat(trad,100))
        tnumax =  tnumax.reshape(ndata,100)
        if return_samples:
            return tnumax
        return np.median(tnumax, axis=1)

def Kmag_to_lum(Kmag, JK, parallax, AK=None, Mbol_sun=4.67):
    """
    convert apparent K mag, J-K colour and parallax into luminosity
    INPUT:
        Kmag - apparent K band magnitude
        JK - J-K colour
        parallax - parallax in mas
        AK - extinction in K band
        Mbol_sun - the solar bolometric magnitude
    OUTPUT:
        luminosity in L_sun
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    BCK = polybcs.BCK_from_JK(JK)
    if AK is None:
        MK = Kmag-(5*np.log10(1000/parallax)-5)
    else:
        MK = Kmag -(5*np.log10(1000/parallax)-5) - AK
    Mbol = BCK+MK
    lum = 10**(0.4*(Mbol_sun-Mbol))
    return lum

def prob(snr, numax, T, pfalse):
    """
    Probability of asteroseismic detection passing false alarm test
    INPUT:
        snr - SNR_tot for a given observation
        numax - corresponding to the observed star
        T - observation length in days
    OUTPUT:
        detection probability
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    env_width = 0.66 * numax**0.88
    env_width[numax>100.] = numax[numax>100.]/2.
    tlen = T*24*60*60 #T in seconds
    bw=1e6/tlen #bin width in uHz
    nbins= 2*env_width//bw #number of independent freq. bins
    pdet = 1-pfalse
    snrthresh = chi2.ppf(pdet,2.*nbins)/(2.*nbins)-1.0
    return chi2.sf((snrthresh+1.0) / (snr+1.0)*2.0*nbins, 2.*nbins)

def SNR_tot(G, BP, RP, J, H, K, lum, mass, teff, rad, numax, s=1., deltaT=1550, Amax_sun=2.5, obs='kepler-sc', D=1):
    """
    predicted S/N for a given set of parameters
    INPUT:
        mag - relevant magnitude e.g. V or Kp for kepler...
        lum - luminosity in Lsun
        mass - stellar mass in Msun
        teff - effective temperature in K
        rad - stellar radius in Rsun
        s - parameter for Amax, defaults to 1.
        deltaT - free parameter in beta (fit to data, e.g. Chaplin et al. 2011)
        Amax_sun - solar maximum oscillation intensity Amplitude
        obs - the observing mode of the data (e.g. 'kepler-sc' for short-cadence kepler)
    OUTPUT:
        SNR  - the signal-to-noise ratio
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    if obs.lower() == 'kepler-sc':
        cadence = 58.85 #in s
        inst = 'kepler'
    if obs.lower() == 'kepler-lc':
        cadence = 29.4*60 #in s
        inst = 'kepler'
    if obs.lower() == 'tess-ffi':
        cadence = 30.*60. #in s
        inst = 'tess'
    nu_nyq = 1e6/(2*cadence) #in uHz
    tP_tot = P_tot(lum, mass, teff, rad, s=s, deltaT=deltaT, Amax_sun=Amax_sun, nu_nyq=nu_nyq, D=D)
    tB_tot = B_tot(G, BP, RP, J, H, K, lum, mass, teff, rad, numax, cadence, inst=inst, nu_nyq=nu_nyq, s=s, deltaT=deltaT, Amax_sun=Amax_sun)
    return tP_tot/tB_tot

def J_K_Teff(JK, FeH=None, err=None):
    """
    Teff from J-K colour based on Gonzalez Hernandez and Bonifacio (2009)
    INPUT:
        JK - J-K colour
        FeH - the [Fe/H] for each entry
        err - error on JK (optional)
    OUTPUT:
        T_eff - the effective temperature
        T_eff_err - error on T_eff
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    if FeH is None:
        #include a prior on feh? for now just assume solar
        theff = 0.6524 + 0.5813*JK + 0.1225*JK**2.
        if err is not None:
            b2ck=(0.5813+2*0.1225*JK)
            a = (5040*b2ck/(0.6524+JK*b2ck)**2)**2
            tefferr = np.sqrt(a*err**2)
    else:
        theff = 0.6524 + 0.5813*JK + 0.1225*JK**2. - 0.0646*JK*FeH + 0.0370*FeH + 0.0016*FeH**2.
    if err is not None:
        return 5040/theff, tefferr
    return 5040/theff

def numax(mass,teff,rad):
    """
    nu_max from scaling relations
    INPUT:
        mass - stellar mass in Msun
        teff - Teff in K
        rad - stellar radius in Rsun
    OUTPUT:
        numax - numax from scaling relations in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    return numax_sun*mass*(teff/teff_sun)**-0.5*rad**-2.

def dnu(mass,rad):
    """
    delta nu from scaling relations
    INPUT:
        mass - stellar mass in Msun
        rad - stellar radius in Rsun
    OUTPUT:
        deltanu - delta nu based on scaling relations in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    return dnu_sun*(mass*rad**-3.)**0.5

def luminosity(rad,teff):
    """
    luminosity in L_sun from scaling relations
    INPUT:
        rad - stellar radius in Rsun
        teff - Teff in K
    OUTPUT:
        lum - luminosity in L_sun from scaling relations
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    return rad**2*(teff/teff_sun)**4.

def teffred(lum):
    """
    T_red, temperature on red edge of Delta Scuti instability strip (see Chaplin+ 2011)
    """
    return teffred_sun*(lum)**-0.093

def beta(teffred, teff, deltaT=1550):
    """
    beta parameter (for Amax) taken from eq. (9) of Chaplin+ (2011)
    INPUT:
        teffred - temp on red edge of delta scuti instability strip
        teff - T_eff in K
    OUTPUT:
        beta - beta parameter
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    if hasattr(teff, '__iter__'):
        out= np.zeros(len(teff))
        out[teff >= teffred] = 0.
        out[teff < teffred] = 1 - np.exp(-(teffred[teff < teffred]-teff[teff < teffred])/deltaT)
    else:
        if teff >= teffred:
            out = 0.
        else:
            out = 1 - np.exp(-(teffred-teff)/deltaT)
    return out

def A_max(rad,lum,mass,teff,s=1., deltaT=1550, Amax_sun=2.5):
    """
    maximum oscillation intensity amplitude from (7) of Chaplin+(2011)
    INPUT:
        lum - luminosity in Lsun
        mass - mass in Msun
        teff - teff in K
        s - parameter, default 1.
        deltaT - parameter for teffred, default 1550 (Chaplin+ 2011)
    OUTPUT:
        A_max - maxmum oscillation intensity
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tteffred = teffred(lum)
    tbeta = beta(tteffred, teff, deltaT=deltaT)
    return  2.5*tbeta*(lum/mass)*(teff/teff_sun)**(-2.0);#0.85*2.5*tbeta*(rad)**2*(teff/teff_sun)**(0.5)#0.85*2.5*tbeta*(rad**1.85)*((teff/teff_sun)**0.57)

def P_tot(lum, mass, teff, rad, s=1., deltaT=1550, Amax_sun=2.5, nu_nyq=8486, D=1.):
    """
    total mean power in envelope
    INPUT:
        lum - luminosity in Lsun
        mass - mass in Msun
        teff - teff in K
        rad - radius in R_sun
        s - parameter, default 1.
        deltaT - parameter for teffred, default 1550 (Chaplin+ 2011)
    OUTPUT:
        P_tot - mean power in envelope (ppm^2/uHz)?
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tA_max = A_max(rad, lum, mass,teff, s=s, deltaT=deltaT, Amax_sun=Amax_sun)
    tnumax = numax(mass, teff, rad)
    #tdnu = dnu(mass, rad)
    eta = np.sin(np.pi/2.*(tnumax/nu_nyq))/(np.pi/2.*(tnumax/nu_nyq))
    env_width = 0.66 * tnumax**0.88
    env_width[tnumax>100.] = tnumax[tnumax>100.]/2.
    tdnu = dnu_sun*(rad**-1.42)*((teff/teff_sun)**0.71)
    return 0.5*2.94*tA_max**2*(tnumax/tdnu)*(np.sinc(np.pi/2.0*(tnumax/nu_nyq)))**2*(D**-2)#0.5*2.94*tA_max**2.*(((2*env_width)/tdnu)*eta**2)

def b_inst(G, BP, RP, J, H, K, cadence, inst='kepler'):
    """
    instrumental background noise for given observation (ppm?)
    INPUT:
        mag - relevant apparent magnitude (e.g Kep. Mag, V, for Kepler, T for Tess)
        numax - numax in uHz
        cadence - cadence in seconds (integration time for tess)
        inst - the instrument used ('kepler', 'corot', 'tess')
    OUTPUT:
        b_inst - background noise in ppm
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    if inst.lower() == 'kepler':
        return noise.kepler_noise_model(G, BP, RP, cadence)
    if inst.lower() == 'tess':
        return noise.tess_noise_model(G, BP, RP, cadence)


def P_gran(tnumax, nu_nyq, ret_eta=False, D=1.):
    """
    granulation power at numax
    INPUT:
        tnumax - numax in uHz
        nu_nyq - the nyquist frequency in uHz
    OUTPUT:
        Pgran - power due to granulation at the envelope
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """

    # modelF = np.zeros(len(tnumax))
    # for i in range(len(tnumax)):
    #     a = 0.85*3382*tnumax[i]**(-0.609)
    #     b  = np.array([0.317*tnumax[i]**(0.970), 0.948*tnumax[i]**0.992])
    #     modelF[i] = np.sum(((2*np.sqrt(2)/np.pi)*a**2/b)/(1+(tnumax[i]/b)**4))
    Pgran = 0.2*(tnumax/numax_sun)**(-2.0)*(D**-2)#np.sinc(np.pi/2.*(tnumax/nu_nyq))**2*(D**-2)*modelF
    return Pgran

def B_tot(G, BP, RP, J, H, K, lum, mass, teff, rad, tnumax, cadence, inst='kepler', nu_nyq=8486,  s=1., deltaT=1550, Amax_sun=2.5):
    """
    total underlying background power
    INPUT:
        mag - relevant apparent magnitude (e.g Kep. Mag, V, for Kepler)
        numax - numax in uHz
        cadence - cadence in seconds
        inst - the instrument used ('kepler', 'corot', 'tess')
    OUTPUT:
        B_tot - background power in ppm^2/uHz?
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tb_inst = b_inst(G, BP, RP, J, H, K, cadence, inst=inst)
    tP_gran = P_gran(tnumax, nu_nyq,)# ret_eta=True)
    #pgranalias = np.zeros(len(tnumax))
    #pgranalias[tnumax > nu_nyq] = P_gran((nu_nyq - (tnumax[tnumax > nu_nyq] - nu_nyq)), nu_nyq)
    #pgranalias[tnumax <= nu_nyq] = P_gran((nu_nyq + (nu_nyq - tnumax[tnumax <= nu_nyq])), nu_nyq)
    #totpgran = tP_gran + pgranalias
    tA_max = A_max(rad, lum, mass,teff, s=s, deltaT=deltaT, Amax_sun=Amax_sun)
    #env_width = 0.66 * tnumax**0.88
    #env_width[tnumax>100.] = tnumax[tnumax>100.]/2.
    tdnu = dnu_sun*(rad**-1.42)*((teff/teff_sun)**0.71)
    return ((2.e-6*tb_inst**2*cadence+tP_gran)*tnumax)#*2*env_width)#*1e-6 #1e-6 factor??
