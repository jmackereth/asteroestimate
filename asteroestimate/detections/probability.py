import numpy as np
from scipy.stats import chi2
from scipy.interpolate import interp1d
import asteroestimate.detections.noise as noise
import asteroestimate.bolometric.polynomial as polybcs

numax_sun = 3150 # uHz
dnu_sun = 135.1 # uHz
teff_sun = 5777 # K
taugran_sun = 210 # s
teffred_sun = 8907 # K

def from_phot(G, BP, RP, J, H, K, parallax, s=1., deltaT=1550., Amax_sun=2.5, obs='kepler-sc', T=30., pfalse=0.01, mass=1., AK=None, numax_limit=None, return_SNR=False):
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
    tlum = Kmag_to_lum(K, J-K, parallax, AK=AK, Mbol_sun=4.67) #luminosity in Lsun
    tteff = J_K_Teff(J-K) #teff in K
    trad = np.sqrt(tlum/(tteff/teff_sun)**4)
    if isinstance(mass, (int, float,np.float32,np.float64)):
        tmass = mass
        tnumax = numax(tmass, tteff, trad)
        snrtot = SNR_tot(G, BP, RP, J, H, K, tlum, tmass, tteff, trad, s=s, deltaT=deltaT, Amax_sun=Amax_sun, obs=obs)
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
        snrtots = SNR_tot(np.repeat(G,100),np.repeat(BP,100),np.repeat(RP,100),np.repeat(J,100),np.repeat(H,100),np.repeat(K,100),
                          np.repeat(tlum,100), msamples, np.repeat(tteff,100), np.repeat(trad,100),
                          s=s, deltaT=deltaT, Amax_sun=Amax_sun, obs=obs)
        tnumax = numax(msamples, np.repeat(tteff,100), np.repeat(trad,100))
        probs = prob(snrtots,tnumax,np.repeat(T,100),pfalse)
        probs = probs.reshape(ndata,100)
        probs = np.median(probs, axis=1)
        if numax_limit is not None:
            probs[np.median(tnumax.reshape(ndata,100), axis=1) < numax_limit] = 0.
        if return_SNR:
            return probs, np.median(snrtots.reshape(ndata,100),axis=1)
        return probs

def numax_from_JHK(J, H, K, parallax, mass=1., return_samples=False, return_lum=False, AK=None):
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
    tteff = J_K_Teff(J-K) #teff in K
    tteff /= teff_sun
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
    tlen=T*86400.0 #T in seconds
    bw=1e6/tlen #bin width in uHz
    nbins= 2*env_width//bw #number of independent freq. bins
    pdet = 1-pfalse
    snrthresh = chi2.ppf(pdet,2.*nbins)/(2.*nbins)-1.0
    return chi2.sf((snrthresh+1.0) / (snr+1.0)*2.0*nbins, 2.*nbins)

def SNR_tot(G, BP, RP, J, H, K, lum, mass, teff, rad, s=1., deltaT=1550, Amax_sun=2.5, obs='kepler-sc'):
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
        nu_nyq = 1e6/(2*cadence) #in uHz
    if obs.lower() == 'kepler-lc':
        cadence = 29.4*60 #in s
        inst = 'kepler'
        nu_nyq = 1e6/(2*cadence) #in uHz
    if obs.lower() == 'tess-ffi':
        cadence = 30.*60. #in s
        inst = 'tess'
        nu_nyq = 1e6/(2*cadence) #in uHz
    tP_tot = P_tot(lum, mass, teff, rad, s=s, deltaT=deltaT, Amax_sun=Amax_sun, nu_nyq=nu_nyq)
    tnumax = numax(mass,teff,rad)
    tB_tot = B_tot(G, BP, RP, J, H, K, lum, mass, teff, rad, tnumax, cadence, inst=inst, nu_nyq=nu_nyq, s=s, deltaT=deltaT, Amax_sun=Amax_sun)
    return tB_tot

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

def A_max(rad,lum,teff,s=1., deltaT=1550, Amax_sun=2.5):
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
    return  0.85*2.5*tbeta*(rad**1.85)*((teff/teff_sun)**0.57)

def P_tot(lum, mass, teff, rad, s=1., deltaT=1550, Amax_sun=2.5, nu_nyq=8486):
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
    tA_max = A_max(rad, lum, teff, s=s, deltaT=deltaT, Amax_sun=Amax_sun)
    tnumax = numax(mass, teff, rad)
    #tdnu = dnu(mass, rad)
    eta = np.sin(np.pi/2.*(tnumax/nu_nyq))/(np.pi/2.*(tnumax/nu_nyq))
    env_width = 0.66 * tnumax**0.88
    env_width[tnumax>100.] = tnumax[tnumax>100.]/2.
    tdnu = dnu_sun*(rad**-1.42)*((teff/teff_sun)**0.71)
    return 0.5*2.94*tA_max**2.*(((2*env_width)/tdnu)*eta**2)

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


def P_gran(tnumax, nu_nyq, ret_eta=False):
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
    a_nomass = 0.85 * 3382*tnumax**-0.609 # multiply by 0.85 to convert to redder TESS bandpass.
    b1 = 0.317 * tnumax**0.970
    b2 = 0.948 * tnumax**0.992
    Pgran = (((2*np.sqrt(2))/np.pi) * (a_nomass**2/b1) / (1 + ((tnumax/b1)**4)) + ((2*np.sqrt(2))/np.pi) * (a_nomass**2/b2) / (1 + ((tnumax/b2)**4)))
    eta = np.sinc(tnumax/(2*nu_nyq))
    Pgran = Pgran *  eta**2
    if ret_eta:
        return Pgran, eta
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
    tP_gran, teta = P_gran(tnumax, nu_nyq, ret_eta=True)
    pgranalias = np.zeros(len(tnumax))
    pgranalias[tnumax > nu_nyq] = P_gran((nu_nyq - (tnumax[tnumax > nu_nyq] - nu_nyq)), nu_nyq)
    pgranalias[tnumax <= nu_nyq] = P_gran((nu_nyq + (nu_nyq - tnumax[tnumax <= nu_nyq])), nu_nyq)
    totpgran = tP_gran + pgranalias
    tA_max = A_max(rad, lum, teff, s=s, deltaT=deltaT, Amax_sun=Amax_sun)
    env_width = 0.66 * tnumax**0.88
    env_width[tnumax>100.] = tnumax[tnumax>100.]/2.
    tdnu = dnu_sun*(rad**-1.42)*((teff/teff_sun)**0.71)
    ptot = 0.5*2.94*tA_max**2.*((2.*env_width)/tdnu)*teta**2.
    return ptot/((tb_inst+tP_gran)*2*env_width)#*1e-6 #1e-6 factor??
