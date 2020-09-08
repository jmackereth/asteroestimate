import numpy as np

def kepler_noise_model(G, BP, RP, cadence):
    """
    return the ppm/hr noise for Kepler
    INPUT:
        G - G band magnitude
        BP - G_BP magnitude (not used for kepler!)
        RP - G_RP magnitude (not used for kepler!)
        cadence - the integration time
    OUTPUT:
        noise - the instrument noise in ppm/hour
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    #kepler noise model (Gilliland+ 2010)... have to assume V ~ G
    mag  =  G
    c = 1.28 * 10**(0.4*(12-mag)+7)
    sigma =  1e6/c*(c+9.5e5*(14/mag)**5)**0.5
    return 2e-6*sigma**2*cadence

def tess_noise_model(G, BP, RP, cadence):
    """
    return the ppm/hr noise for TESS
    INPUT:
        G - G band magnitude
        BP - G_BP magnitude (not used for kepler!)
        RP - G_RP magnitude (not used for kepler!)
        cadence - the integration time
    OUTPUT:
        noise - the instrument noise in ppm
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    #Stassun model
    color = BP-RP
    mag = G
    integration = cadence/(60.) #cadence back to mins!
    tessmag = mag - 0.00522555*color**3 + 0.0891337*color**2 - 0.633923*color + 0.0324473
    #return get_oneSigmaNoise(integration, tessmag)
    return noise_fit_27min(G)


def noise_fit_27min(G):
    coeff= np.array([ 6.20408891e-06,  5.97152412e-05,  2.54251067e-04, -2.53740192e-03, -3.57921614e-02,  1.44013454e+00])
    poly = np.poly1d(coeff)
    return 10**poly(G)

def get_oneHourNoiseLnsigma(TessMag):
        """
        from tessgi/ticgen
        TESS photometric error estimate [ppm] based on
        magnitude and Eq. on bottom of P24 of
        arxiv.org/pdf/1706.00495.pdf

        seems like a fit to curves in other papers...?
        """
        F = 4.73508403525e-5
        E = -0.0022308015894
        D = 0.0395908321369
        C = -0.285041632435
        B = 0.850021465753
        lnA = 3.29685004771

        return (lnA + B * TessMag + C * TessMag**2 + D * TessMag**3 +
                E * TessMag**4 + F * TessMag**5)

def get_oneSigmaNoise(exp_time, TessMag):
    """ from tessgi/ticgen """
    onesig = (np.exp(get_oneHourNoiseLnsigma(TessMag))/np.sqrt(exp_time / 60.))
    if hasattr(onesig, '__iter__'):
        onesig[onesig < 60] = 60.
    elif onesig < 60:
        onesig = 60.
    return onesig
