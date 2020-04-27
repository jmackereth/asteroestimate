import numpy as np

def BCv_from_teff(teff):
    """  from F Pijpers 2003. BCv values from Flower 1996 polynomials presented in Torres 2010
        taken from MathewSchofield/ATL_public """
    lteff = np.log10(teff)
    BCv = np.zeros(len(teff))

    BCv[lteff<3.70] = (-0.190537291496456*10.0**5) + \
    (0.155144866764412*10.0**5*lteff[lteff<3.70]) + \
    (-0.421278819301717*10.0**4.0*lteff[lteff<3.70]**2.0) + \
    (0.381476328422343*10.0**3*lteff[lteff<3.70]**3.0)

    BCv[(3.70<lteff) & (lteff<3.90)] = (-0.370510203809015*10.0**5) + \
    (0.385672629965804*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]) + \
    (-0.150651486316025*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]**2.0) + \
    (0.261724637119416*10.0**4*lteff[(3.70<lteff) & (lteff<3.90)]**3.0) + \
    (-0.170623810323864*10.0**3*lteff[(3.70<lteff) & (lteff<3.90)]**4.0)

    BCv[lteff>3.90] = (-0.118115450538963*10.0**6) + \
    (0.137145973583929*10.0**6*lteff[lteff > 3.90]) + \
    (-0.636233812100225*10.0**5*lteff[lteff > 3.90]**2.0) + \
    (0.147412923562646*10.0**5*lteff[lteff > 3.90]**3.0) + \
    (-0.170587278406872*10.0**4*lteff[lteff > 3.90]**4.0) + \
    (0.788731721804990*10.0**2*lteff[lteff > 3.90]**5.0)
    return BCv

def BCG_from_teff(teff):
    """ taken from https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu8par/sec_cu8par_process/ssec_cu8par_process_flame.html"""
    nteff = teff-teff_sun
    out = np.zeros(len(teff))

    out[teff < 4000] = 1.749 +\
                      (1.977e-3*nteff[teff < 4000]) +\
                      (3.737e-7*nteff[teff < 4000]**2) +\
                      (-8.966e-11*nteff[teff < 4000]**3) +\
                      (-4.183e-14*nteff[teff < 4000]**4)

    out[teff >= 4000] = 6e-2 +\
                       (6.731e-5*nteff[teff >= 4000]) +\
                       (-6.647e-8*nteff[teff >= 4000]**2) +\
                       (2.859e-11*nteff[teff >= 4000]**3) +\
                       (-7.197e-15*nteff[teff >= 4000]**4)

    return out

def BCK_from_JK(JK):
    """based on a simple fit to Houdashelt+2000 Table 5 """
    coeff = np.array([-1.27123055,  3.69172478,  0.11070501])
    poly = np.poly1d(coeff)
    out = poly(JK)
    return out
