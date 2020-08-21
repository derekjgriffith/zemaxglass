#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import os, glob, sys
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.transforms import offset_copy
import colorsys
from cycler import cycler
import pdb
from scipy.optimize import curve_fit, least_squares  # e.g. for fitting Buchdahl dispersion functions
clear_output_possible = True
import warnings
try:
    from IPython.display import clear_output
except ImportError as e:
    clear_output_possible = False
try:
    import ref_index
except ImportError as e:
    warnings.warn('Some air refractive index computations may use the Python module ref_index. Install with "pip install ref_index".')
import pandas as pd
import re
from datetime import datetime, timedelta
import copy



"""
This file contains a set of utilities for reading Zemax glass (*.agf) files, analyzing glass
properties, and displaying glass data.

It also implements some methods of glass selection for achromatization and athermalization
of optical systems. 

See LICENSE.txt for a description of the MIT/X license for this file.

The module has dependencies on numpy, matplotlib, scipy, pandas and the ref_index module.
"""

__authors__ = 'Nathan Hagen, Derek Griffith'
__license__ = 'MIT/X11 License'
__contact__ = 'Nathan Hagen <and.the.light.shattered@gmail.com>'

# Define some spectral/line wavelengths commonly used in this context (all in nm)
# Source Schott technical note TIE 29.
# Note that Zemax uses units of microns when specifying wavelength
wv_Hg_IR3 = 2325.42  # Shortwave infrared mercury line Hg
wv_Hg_IR2 = 1970.09  # Shortwave infrared mercury line Hg
wv_Hg_IR1 = 1529.582  # Shortwave infrared mercury line Hg
wv_NdYAG = 1064.0  # Neodymium glass laser Nd
wv_t = 1013.98  # Shortwave infrared Hg line
wv_s = 852.11  # Near infrared Cs line 
wv_r = 706.5188  # Red He line
wv_C = 656.2725  # Red H line
wv_C_prime = 643.8469  # Red Cd line
wv_HeNe = 632.8  # Helium-Neon Laser
wv_D = 589.2938  # Orange Na line
wv_d = 587.5618  # Yellow He line
wv_e = 546.074  # e green mercury line Hg
wv_F = 486.1327  # F blue hydrogen line H
wv_F_prime = 479.9914   # F' blue cadmium line Cd
wv_g = 435.8343  # g blue mercury line Hg
wv_h = 404.6561  # h violet mercury line Hg
wv_i = 365.0146  # i ultraviolet mercury line Hg
wv_Hg_UV1 = 334.1478  # ultraviolet mercury line Hg
wv_Hg_UV2 = 312.5663  # ultraviolet mercury line Hg
wv_Hg_UV3 = 296.7278  # ultraviolet mercury line Hg
wv_Hg_UV4 = 280.4  # ultraviolet mercury line Hg
wv_Hg_UV5 = 248.3  # ultraviolet mercury line Hg
wv_Hg = np.array([wv_Hg_IR3, wv_Hg_IR2, wv_Hg_IR1, wv_e, wv_g, wv_h, wv_i, wv_Hg_UV1, wv_Hg_UV2, wv_Hg_UV3, wv_Hg_UV4, wv_Hg_UV5])

# Define "named" lines as a dict as well
wv_dict = {"NdYAG":wv_NdYAG, "t": wv_t, "s": wv_s, "r": wv_r, "C": wv_C, "C'": wv_C_prime, "HeNe": wv_HeNe, "D": wv_D, "d":wv_d,
           "e": wv_e, "F": wv_F, "F'": wv_F_prime, "g": wv_g, "h":wv_h, "i": wv_i}

# Simple text-based progress bar, used during long computations
def update_progress(progress, bar_length):
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)           

# Catalog folders, original and updated to circa Feb 2020
agfdir = os.path.dirname(os.path.abspath(__file__)) + '/AGF_files/'
agfdir202002 = os.path.dirname(os.path.abspath(__file__)) + '/AGF_files/202002/'

# Number of coefficients for vairous Zemax dispersion relation formulae 
num_coeff = [6, 6, 7, 5, 3, 8, 4, 4, 5, 8, 10, 8, 9]

def zemax_dispersion_formula(wv, dispform, coefficients):
    """
    Calculate catalog material refractive indices according to the various dispersion formulae defined in the Zemax manual.
    For materials defined in Zemax glass catalogues, the returned indices will be relative to air at standard
    temperature and pressure (20C and 1 atmosphere). The wavelengths are then also assumed to be
    provided in air at the same conditions.

    Parameters
    ----------
    wv : list or array of float
        Wavelengths for which to perform the calculation. In air at the standard pressure and temperature. 
        If any given wavelength is above 100.0 it is assumed to be in nm. Otherwise wavelength is assumed to be microns.
    dispform : int
        The index of the formula in the order provided in the Zemax manual and as defined in the .agf file format.
        Range is 1 to 13. ValueError is thrown if not one of these.
        1 : Schott with 6 coefficients
        2 : Sellmeier 1 with 6 coefficients
        3 : Herzberger with 7 coefficients
        4 : Sellmeier 2 with 5 coefficients
        5 : Conrady with 3 coefficients
        6 : Sellmeier 3 with 8 coefficients
        7 : Handbook of Optics 1 with 4 coefficients
        8 : Handbook of Optics 2 with 4 coefficients
        9 : Sellmeier 4 with 5 coefficients
        10: Extended 1 with 8 coefficients
        11: Sellmeier 5 with 10 coefficients
        12: Extended 2 with 8 coefficients
        13: Extended 3 with 9 coefficients
    coefficients : list or array of float
        Coefficents of the dispersion formula. Number of coefficients depends on the formula.
    """
    w = np.asarray(wv, dtype=np.float)
    # If wavelength is above 100.0 assume nm and divide by 1000, otherwise assume microns and no scaling is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) / 1000.0 + np.asarray(wv <= 100.0, dtype=np.float)
    w *= unit_scaling  # Zemax formulae assume microns
    cd = coefficients
    if (dispform == 1):  ## Schott
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 2):  ## Sellmeier1
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + (cd[4] * w**2 / (w**2 - cd[5]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 3):  ## Herzberger
        L = 1.0 / (w**2 - 0.028)
        indices = cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + (cd[4] * w**4) + (cd[5] * w**6)
    elif (dispform == 4):  ## Sellmeier2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + (cd[3] * w**2 / (w**2 - (cd[4])**2))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 5):  ## Conrady
        indices = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
    elif (dispform == 6):  ## Sellmeier3
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                        (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 7):  ## HandbookOfOptics1
        formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 8):  ## HandbookOfOptics2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 9):  ## Sellmeier4
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + (cd[3] * w**2 / (w**2 - cd[4]))
        indices = np.sqrt(formula_rhs)
    elif (dispform == 10):  ## Extended1
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                        (cd[5] * w**-8) + (cd[6] * w**-10) + (cd[7] * w**-12)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 11):  ## Sellmeier5
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                        (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                        (cd[8] * w**2 / (w**2 - cd[9]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 12):  ## Extended2
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                        (cd[5] * w**-8) + (cd[6] * w**4) + (cd[7] * w**6)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 13):  ## Extended3
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**4) + (cd[3] * w**-2) + (cd[4] * w**-4) + \
                        (cd[5] * w**-6) + (cd[6] * w**-8) + (cd[7] * w**-10) + (cd[8] * w**-12)
        indices = np.sqrt(formula_rhs)
    else:
        raise ValueError('Dispersion formula #' + str(dispform) + ' is not a valid choice.')
    return indices

def air_index_kohlrausch(wv, T, P, rh=50.0):
    '''
    Compute the refractive index of air using the Kohlrausch formula.

    Parameters
    ----------
    wv : float or array of float
        Wavelengths. If any wavelength exceeds 100, it is assumed to be in nm.
        If wavelength is below 100, it is assumed to be in units of microns
    T : float
        Temperature in degrees Celcius.
    P : float
        Absolute air pressure in Pa. One atmosphere is 101325 Pa = 101.325 kPa.
        If the pressure is given as zero, refractive indices of 1 are returned for all wavelengths.
    rh : anything
        Dummy input - it does nothing.

    Returns
    -------
    indices : array of float
        Air refractive indices for given wavelengths at given temperature and relative pressure.
    '''
    w = np.asarray(wv, dtype=np.float)
    if P==0.0:
        return np.atleast_1d(np.ones(w.shape))    
    # If wavelength is above 100.0 assume nm and divide by 1000, otherwise assume microns and no scaling is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) / 1000.0 + np.asarray(wv <= 100.0, dtype=np.float)
    w *= unit_scaling  # Kohlrausch formula needs microns    
    n_ref = 1.0 + ((6432.8 + ((2949810.0 * w**2) / (146.0 * w**2 - 1.0)) + ((25540.0 * w**2) / (41.0 * w**2 - 1.0))) * 1.0e-8)
    indices = 1.0 + ((n_ref - 1.0) / (1.0 + (T - 15.0) * 3.4785e-3)) * (P/101325.0)
    return np.atleast_1d(indices)

def air_index_ciddor(wv, T, P, rh=50.0, co2=450.0, warn=False):
    '''
    Compute refractive index of air using the Ciddor equation.
    This is a thin wrapper around the ref_index.ciddor() function.

    Parameters
    ----------
    wv : float or array of float
        Wavelengths. If any wavelength exceeds 100, it is assumed to be in nm.
        If wavelength is below 100, it is assumed to be in units of microns
    T : float
        Temperature in degrees Celcius.
    P : float
        Absolute air pressure in Pa. One atmosphere is 101325 Pa = 101.325 kPa.
        If the pressure is given as zero, refractive indices of 1 are returned for all wavelengths.        
    rh : float  
        Relative humidity in percentage. Default is 50%.
    co2 : float
        Carbon dioxide concentration in µmole/mole. The default value
        of 450 should be enough for most purposes. Valid range is from
        0 - 2000 µmole/mole.
    warn : boolean
        Warning is issued if parameters fall outside accept
        range. Accepted range is smaller than the valid ranges
        mentioned above. See module docstring for accepted ranges.
    '''
    w = np.asarray(wv, dtype=np.float)
    if P==0.0:
        return np.atleast_1d(np.ones(w.shape))     
    # If wavelength is above 100.0 assume nm, otherwise assume microns and multiply by 1000.0 is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) + np.asarray(wv <= 100.0, dtype=np.float) * 1000.0   
    w *= unit_scaling  # Ciddor function needs wavelength in nanometres
    indices = ref_index.ciddor(wave=w, t=T, p=P, rh=rh, co2=co2, warn=warn)
    return np.atleast_1d(indices)

def air_index_edlen(wv, T, P, rh=50.0, warn=False):
    '''
    Compute refractive index of air using the Edlen equation.
    This is a thin wrapper around the ref_index.edlen() function.

    Parameters
    ----------
    wv : float or array of float
        Wavelengths. If any wavelength exceeds 100, it is assumed to be in nm.
        If wavelength is below 100, it is assumed to be in units of microns
    T : float
        Temperature in degrees Celcius.
    P : float
        Absolute air pressure in Pa. One atmosphere is 101325 Pa = 101.325 kPa.
        If the pressure is given as zero, refractive indices of 1 are returned for all wavelengths.        
    rh : float  
        Relative humidity in percentage. Default is 50%.
    warn : boolean
        Warning is issued if parameters fall outside accept
        range. Accepted range is smaller than the valid ranges
        mentioned above. See module docstring for accepted ranges.
    '''
    w = np.asarray(wv, dtype=np.float)
    if P==0.0:
        return np.atleast_1d(np.ones(w.shape))     
    # If wavelength is above 100.0 assume nm, otherwise assume microns and multiply by 1000.0 is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) + np.asarray(wv <= 100.0, dtype=np.float) * 1000.0   
    w *= unit_scaling  # Edlen function needs wavelength in nanometres
    indices = ref_index.edlen(wave=w, t=T, p=P, rh=rh, warn=warn)
    return np.atleast_1d(indices)

def buchdahl_omega(wv, wv_center, alpha):
    r"""
    Calculate the Buchdahl dispersion relation spectral coordinates (omega) for given wavelengths, central wavelength and alpha parameter.

    Parameters
    ----------
    wv : array of float
        The wavelengths at which to calculate the Buchdahl omega dispersion relation spectral coordinates.
        If any given wavelength is above 100.0 it is assumed to be in nm. Otherwise wavelength is assumed to be microns.
    wv_center : float
        The central wavelength for the spectral region under consideration. If greater than 100.0, nanometer units assumed,
        otherwise microns.
    alpha : float
        The Buchdahl alpha parameter, generally a constant over a category or even an entire catalog of glasses.
        However, alpha can also be tuned to give optimal Buchdahl dispersion formula fit for a specific glass.

    Returns
    -------
    omega : array of float
        Buchdahl omega spectral coordinates for the given wavelengths, computed as
        $$\\omega=\\frac{\\lambda-\\lambda_0}{1+\\alpha(\\lambda-\\lambda_0)}$$.
    """
    wv = np.asarray(wv, dtype=np.float)
    # If wavelength is above 100.0 assume nm and divide by 1000, otherwise assume microns and no scaling is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) / 1000.0 + np.asarray(wv <= 100.0, dtype=np.float)
    wv *= unit_scaling  # Zemax formulae assume microns
    # Same treatment for central wavelength, if above 100.0 assume nanometres and convert to microns
    if wv_center > 100.0: wv_center /= 1000.0
    omega = (wv - wv_center) / (1.0 + alpha * (wv - wv_center))
    return omega

def buchdahl_model(wv, wv_center, n_center, alpha, *argv):
    r"""
    Compute the refractive indices for the specified Buchdahl dispersion model at the specified wavelengths.

    Parameters
    ----------
    wv : array of float
        The wavelengths $\\lambda$ at which to calculate the Buchdahl omega dispersion relation spectral coordinates.
        If any given wavelength is above 100.0 it is assumed to be in nm. Otherwise wavelength is assumed to be microns.
    wv_center : float
        The central wavelength $\lambda_0$ for the spectral region under consideration. If greater than 100.0, nanometer units assumed,
        otherwise microns.
    n_center : float
        The refractive index $n_0$ at the central wavelength wv_center ($\\lambda_0$).        
    alpha : float
        The Buchdahl $\\alpha$ parameter, generally a constant over a category or even an entire catalog of glasses.
        However, alpha can also be tuned to give optimal Buchdahl dispersion formula fit for a specific glass.
    *argv : float
        Buchdahl dispersion coefficients $\\nu$. Usually at most 3 (third order fit) coefficients.

    Returns
    -------
    indices : Refractive indices computed at the specified wavelengths according to the Buchdahl model
        $$n = n_center + \\nu_1\\omega + \\nu_2\\omega^2 + \\nu_3\\omega^3 + \\ldots$$,
        where
        $$\\omega=\\frac{\\lambda-\\lambda_0}{1+\\alpha(\\lambda-\\lambda_0)}$$.
    """
    omega = buchdahl_omega(wv, wv_center, alpha)
    indices = n_center
    for power, nu in enumerate(argv):
        indices += nu * omega**(power+1.0)
    return indices

def buchdahl_model2(wv, wv_center, n_center, alpha, nu_1, nu_2):
    r"""
    Compute the refractive indices for the specified Buchdahl 2nd order dispersion model at the specified wavelengths.

    Parameters
    ----------
    wv : array of float
        The wavelengths $\\lambda$ at which to calculate the Buchdahl omega dispersion relation spectral coordinates.
        If any given wavelength is above 100.0 it is assumed to be in nm. Otherwise wavelength is assumed to be microns.
    wv_center : float
        The central wavelength $\lambda_0$ for the spectral region under consideration. If greater than 100.0, nanometer units assumed,
        otherwise microns.
    n_center : float
        The refractive index $n_center$ at the central wavelength wv_center ($\\lambda_0$).        
    alpha : float
        The Buchdahl $\\alpha$ parameter, generally a constant over a category or even an entire catalog of glasses.
        However, alpha can also be tuned to give optimal Buchdahl dispersion formula fit for a specific glass.
    nu_1 : float
        Buchdahl dispersion coefficient $\\nu_1$. 
    nu_2 : float
        Buchdahl dispersion coefficient $\\nu_2$.

    Returns
    -------
    indices : Refractive indices computed at the specified wavelengths according to the Buchdahl 2nd order model
        $$n = n_center + \\nu_1\\omega + \\nu_2\\omega^2$$,
        where
        $$\\omega=\\frac{\\lambda-\\lambda_0}{1+\\alpha(\\lambda-\\lambda_0)}$$.
    """
    omega = buchdahl_omega(wv, wv_center, alpha)
    indices = n_center + nu_1 * omega + nu_2 * omega**2.0
    return indices

def buchdahl_model3(wv, wv_center, n_center, alpha, nu_1, nu_2, nu_3):
    r"""
    Compute the refractive indices for the specified Buchdahl 3rd order dispersion model at the specified wavelengths.

    Parameters
    ----------
    wv : array of float
        The wavelengths $\\lambda$ at which to calculate the Buchdahl omega dispersion relation spectral coordinates.
        If any given wavelength is above 100.0 it is assumed to be in nm. Otherwise wavelength is assumed to be microns.
    wv_center : float
        The central wavelength $\lambda_0$ for the spectral region under consideration. If greater than 100.0, nanometer units assumed,
        otherwise microns.
    n_center : float
        The refractive index $n_0$ at the central wavelength wv_center ($\\lambda_0$).        
    alpha : float
        The Buchdahl $\\alpha$ parameter, generally a constant over a category or even an entire catalog of glasses.
        However, alpha can also be tuned to give optimal Buchdahl dispersion formula fit for a specific glass.
    nu_1 : float
        Buchdahl dispersion coefficient $\\nu_1$. 
    nu_2 : float
        Buchdahl dispersion coefficient $\\nu_2$.
    nu_3 : float
        Buchdahl dispersion coefficient $\\nu_3$.        

    Returns
    -------
    indices : Refractive indices computed at the specified wavelengths according to the Buchdahl 3rd order model
        $$n = n_center + \\nu_1\\omega + \\nu_2\\omega^2 + \\nu_3\\omega^3$$,
        where
        $$\\omega=\\frac{\\lambda-\\lambda_0}{1+\\alpha(\\lambda-\\lambda_0)}$$.
    """
    omega = buchdahl_omega(wv, wv_center, alpha)
    indices = n_center + nu_1 * omega + nu_2 * omega**2.0 + nu_3 * omega**3.0
    return indices             

def buchdahl_fit(wv, indices, wv_center, n_center, alpha, order=3):
    r"""
    Determine a Buchdahl dispersion function fit to a set of refractive indices at a given set of wavelengths.
    The fit is performed using the non-linear least squares method. The Buchdahl alpha parameter is
    considered fixed in this fitting procedure and must be provided. 
    See buchdahl_fit_alpha() which allows alpha to vary as well.

    Parameters
    ----------
    wv : array of float
        Wavelengths at which the refractive index data is provided. Units assumed nm if wv > 100.0.
    indices : array of float, same length as wv
        Refractive indices at the specified wavelengths.
    wv_center : float
        Center wavelength. Assumed nm if above 100.0, otherwise micron.
    n_center : float
        Refractive index at the center wavelength.
    alpha : float
        The Buchdahl alpha parameter.
    order : int
        The Buchdahl polynomial order. Either 2 or 3, default is 3rd order.

    Returns
    -------
    fit_parms : array of float
        The best fit parameters in order of n_center, nu_1, nu_2 and (if order is 3) nu_3.
    """
    wv = np.asarray(wv, dtype=np.float)
    # If wavelength is above 100.0 assume nm and divide by 1000, otherwise assume microns and no scaling is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) / 1000.0 + np.asarray(wv <= 100.0, dtype=np.float)
    wv *= unit_scaling  # Zemax formulae assume microns
    # Same treatment for central wavelength, if above 100.0 assume nanometres and convert to microns
    if wv_center > 100.0: wv_center /= 1000.0
    if order == 2:
        # The lambda function takes the independent variable (x, the wavelengths) and the parameters we wish to fit.
        # The parameters skipped in the lambda list are then fetched from the local namespace, in the following case, wv_center and alpha (not fitted)
        fit_parms, pcov = curve_fit(lambda x, nu_1, nu_2: buchdahl_model2(x, wv_center, n_center, alpha, nu_1, nu_2), wv, indices)
    elif order == 3:
        fit_parms, pcov = curve_fit(lambda x, nu_1, nu_2, nu_3: buchdahl_model3(x, wv_center, n_center, alpha, nu_1, nu_2, nu_3), wv, indices)
    else:
        raise ValueError('Only 2nd and third order Buchdahl models can be fitted.')
    # Fitted parameters are n_center, nu_1, nu_2 and (if order==3) nu_3    
    return fit_parms

def buchdahl_fit_alpha(wv, indices, wv_center, n_center, order=3):
    r"""
    Determine a Buchdahl dispersion function fit to a set of refractive indices at a given set of wavelengths.
    The fit is performed using the non-linear least squares method. This fit includes fitting of the
    Buchdahl alpha parameter. This will not, except by major coincidence, return the value of alpha
    which provides a linear decline of wavelength with respect to the Buchdahl omega spectral parameter.
    See buchdahl_fit() for fitting the Buchdahl model where the alpha parameter is determined and fixed.

    Parameters
    ----------
    wv : array of float
        Wavelengths at which the refractive index data is provided. Units assumed nm if wv > 100.0.
    indices : array of float, same length as wv
        Refractive indices at the specified wavelengths.
    wv_center : float
        Center wavelength. If above 100.0 assumed nm, otherwise micron units.
    n_center : float
        Refractive index at the center wavelength.
    order : int
        The Buchdahl polynomial order. Either 2 or 3, default is 3rd order.

    Returns
    -------
    fit_parms : array of float
        The best fit parameters in order of alpha, n_center, nu_1, nu_2 and (if order is 3) nu_3.

    See Also
    --------
    buchdahl_fit(), buchdahl_model()
    """
    wv = np.asarray(wv, dtype=np.float)
    # If wavelength is above 100.0 assume nm and divide by 1000, otherwise assume microns and no scaling is performed
    unit_scaling = np.asarray(wv > 100.0, dtype=np.float) / 1000.0 + np.asarray(wv <= 100.0, dtype=np.float)
    wv *= unit_scaling  # Zemax formulae assume microns
    # Same treatment for central wavelength, if above 100.0 assume nanometres and convert to microns
    if wv_center > 100.0: wv_center /= 1000.0
    if order == 2:
        # The lambda function takes the independent variable (x, the wavelengths) and the parameters we wish to fit
        # The parameters skipped in the lambda list are then fetched from the local namespace, in the following case, wv_center
        fit_parms, pcov = curve_fit(lambda x, alpha, nu_1, nu_2: buchdahl_model2(x, wv_center, n_center, alpha, nu_1, nu_2), wv, indices)
    elif order == 3:
        fit_parms, pcov = curve_fit(lambda x, alpha, nu_1, nu_2, nu_3: buchdahl_model3(x, wv_center, n_center, alpha, nu_1, nu_2, nu_3), wv, indices)
    else:
        raise ValueError('Only 2nd and third order Buchdahl models can be fitted.')
    # Fitted parameters are alpha, n_center, nu_1, nu_2 and (if order==3) nu_3
    return fit_parms

def buchdahl_non_linear_error(parms, wv, indices, wv_center, n_center):
    """
    Calculate the deviation of a fitted Buchdahl index vs omega curve from linear.
    This is a helper function to facilitate the use of scipy.least_squares()

    Parameters
    ----------
    parms : list of float
        Main parameters of the Buchdahl fit. parms[0] is alpha, parms[1] is nu_1, parms[2] is nu_2 and
        (if the fit order is 3), parms[3] is nu_3
    wv : array of float
        Wavelengths at which the refractive index data is provided. Units assumed nm if wv > 100.0.
    indices : array of float, same length as wv
        Refractive indices at the specified wavelengths.
    wv_center : float
        Center wavelength. If above 100.0 assumed nm, otherwise micron units.
    n_center : float
        Refractive index at the center wavelength.

    Returns
    -------
    non_linear_error : array of float
        Refractive index linear error metric at each of the wavelengths.
    """
    omega = buchdahl_omega(wv, wv_center, parms[0])  # parms[0] is alpha
    if len(parms) == 3:
        # alpha, nu_1, nu_2
        buch_indices = buchdahl_model2(wv, wv_center, n_center, parms[0], parms[1], parms[2])
    elif len(parms) == 4:
        # alpha, nu_1, nu_2, nu_3
        buch_indices = buchdahl_model3(wv, wv_center, n_center, parms[0], parms[1], parms[2], parms[3])
    lin_fit = np.polyfit(omega, buch_indices, 1)  # Fit straight line
    indices_linear = lin_fit[0] * omega + lin_fit[1]
    non_linear_error = np.sqrt((indices_linear - buch_indices)**2.0 + (indices_linear - indices)**2.0)
    # print('Error :', non_linear_error)
    return non_linear_error


def buchdahl_find_alpha(wv, indices, wv_center, n_center, order=3, gtol=1.0e-9):
    """
    Find the Buchdahl alpha parameter which gives a refractive index versus omega curve that is closest to a straight line.

    Parameters
    ----------
    wv : array of float
        Wavelengths at which the refractive index data is provided. Units assumed nm if wv > 100.0.
    indices : array of float, same length as wv
        Refractive indices at the specified wavelengths.
    wv_center : float
        Center wavelength. If above 100.0 assumed nm, otherwise micron units.
    n_center : float
        Refractive index at the center wavelength.
    order : int
        The Buchdahl polynomial order. Either 2 or 3, default is 3rd order.
    gtol : float
        Controls the convergence accuracy. Default is 1.0e-9. Faster but less accurate convergence 
        will be achieved using larger values. A value of 1.0e-8 will often suffice.  

    Returns
    -------
    optimal_fit_parms : list of float
        The Buchdahl alpha and nu coefficients that provide the best fit to the wavelength refractive
        index data and with the imposed requirement that the index versus Buchdahl curve be close
        to a straight line. The parameters are alpha, nu_1, nu_2 and (if the fit order is 3) nu_3.
    """
    # Run the fit with alpha fitting to get initial values
    # Returned fit parms are alpha, n_center, nu_1, nu_2 and (if order=3) nu_3
    start_fit_parms = buchdahl_fit_alpha(wv, indices, wv_center, n_center, order)
    # The error function in this case is the non-linearity error
    optimal_fit_parms = least_squares(lambda parms, x, y: buchdahl_non_linear_error(parms, x, y, wv_center, n_center), 
                                        x0=start_fit_parms, gtol=gtol, args=(wv, indices))
    return optimal_fit_parms

class ZemaxGlassLibrary(object):
    '''
    ZemaxGlassLibrary is a class to hold all of the information contained in a Zemax-format library of glass catalogs.

    Glass catalogs are in the form of *.agf files, typically given with a vendor name as the filename. The class
    initializer, if given the directory where the catalogs are located, will read them all into a single dictionary
    data structure. The ZemaxLibrary class also gathers together associated methods for manipulating the data, such
    as methods to cull the number of glasses down to a restricted subset, the ability to plot some glass properties
    versus others, the ability to fit different paramatrized dispersion curves to the refractive index data, etc.

    Attributes
    ----------
    dir : str
        The directory where the glass catalog files are stored.
    catalog : float

    Methods
    -------
    pprint
    delete_glasses
    select_glasses
    simplify_schott_catalog
    get_dispersion
    get_polyfit_dispersion
    cull_library
    plot_dispersion
    plot_temperature_dependence
    plot_catalog_property_diagram
    '''
    dispformulas = ['Schott', 'Sellmeier 1', 'Herzberger', 'Sellmeier 2', 'Conrady', 'Sellmeier 3',
                    'Handbook of Optics 1', 'Handbook of Optics 2', 'Sellmeier 4', 'Extended',
                    'Sellmeier 5', 'Extended 2', 'Extended 3']

    def __init__(self, dir=None, wavemin=400.0, wavemax=700.0, nwaves=300, catalog='all', glass_match='.*', glass_exclude='a^',
                sampling_domain='wavelength', degree=3, discard_off_band=True, select_status=None, 
                air_index_function='kohl', debug=False):
        '''
        Initialize the glass library object.

        Parameters
        ----------
        wavemin : float, optional
            The shortest wavelength (nm) in the spectral region of interest.
            If less than a value of 100 wavemin is assumed to have been given in microns and multiplied by 1000.
        wavemax : float, optional
            The longest wavelength (nm) in the spectral region of interest.
            If less than a value of 100 wavemax is assumed to have been given in microns and multiplied by 1000.
        nwaves : float, optional
            The number of wavelength samples to use.
        catalog : str
            The catalog or list of catalogs to look for in "dir".
        glass_match : str
            Regular expression to match. The glass is only included in the returned instance if the glass name
            matches this regular expression. Default is '.*', which matches all glasses. The | operator
            can be used in the RE e.g. to select a number of specific glasses use
            glass_match='N-LAK10|N-SSK5'. Matching is case INsensitive.
        glass_exclude : str
            Regular expression to match for glasses to be excluded. This will override any glasses that have
            been included by default or by glass_match. Default is 'a^', which excludes no glass.  
            Matching is case INsensitive. 
        sampling_domain : str, {'wavelength','wavenumber'}
            Whether to sample the spectrum evenly in wavelength or wavenumber.
        degree : int, optional
            The polynomial degree to use for fitting the dispersion spectrum.
        discard_off_band : boolean
            If set True, will discard glasses where the valid spectral range does not fully cover
            the interval wavemin to wavemax. Default is True.
        select_status : list of int
            Select glasses based on status. One or more of the following status codes:
                0 : Standard
                1 : Preferred
                2 : Obsolete
                3 : Special
                4 : Melt
                Default is all status codes.
        air_index_function : str
            The function used to calculate the refractive index of air when changing pressure.
            One of ['kohl', 'edlen', 'ciddor'] for the Kohlrausch, Edlen and Ciddor equations.
            The default is 'kohl'.
        '''

        self.debug = debug
        self.degree = degree                    ## the degree of polynomial to use when fitting dispersion data
        # Set up the function to calculate the refractive index of air
        self.air_index_function_name = air_index_function
        if air_index_function in ['kohl', 'edlen', 'ciddor']:
            self.air_index_function = {'kohl': air_index_kohlrausch, 'edlen': air_index_edlen, 
                                       'ciddor': air_index_ciddor}[air_index_function]
        else:
            warnings.warn(f'Unknown air refractive index function {air_index_function}. Kohlrausch assumed.')
            self.air_index_function = air_index_kohlrausch
            self.air_index_function_name = 'kohl'
        if wavemin < 100.0:  # Probably given in micron units
            wavemin *= 1000.0
            warnings.warn('Input wavemin is less than 100. Input units of microns assumed and multiplied by 1000 to get units of nm.')
        if wavemax < 100.0:  # Probably given in micron units
            wavemax *= 1000.0
            warnings.warn('Input wavemax is less than 100. Input units of microns assumed and multiplied by 1000 to get units of nm.')  
        #self.basis = basis                     ## the type of basis to use for polynomial fitting ('Taylor','Legendre')
        self.sampling_domain = sampling_domain  ## the domain ('wavelength' or 'wavenumber') in which to evenly sample the data

        if (dir == None):
            dir = os.path.dirname(os.path.abspath(__file__)) + '/AGF_files/'

        self.dir = dir
        self.library, self.cat_comment, self.cat_encoding = read_library(dir, catalog=catalog)
        # Remove glasses where requested wavelength interval is not covered by the valid interval of the dispersion data
        if discard_off_band:
            cat_discard_list = []
            for catalogue in self.library.keys():
                discard_list = []
                for glass in self.library[catalogue].keys():
                    lo_disp_lim = self.library[catalogue][glass]['ld'][0] * 1000.0  # nm
                    hi_disp_lim = self.library[catalogue][glass]['ld'][1] * 1000.0  # nm
                    if lo_disp_lim > wavemin or hi_disp_lim < wavemax: 
                        # print(f'Discarding {catalogue.capitalize()} {glass} (Valid Wavelengths {lo_disp_lim} to {hi_disp_lim} nm)')
                        discard_list.append(glass)
                # Ditch the discarded glasses
                for discarded_glass in discard_list:
                    del self.library[catalogue][discarded_glass]
                # If the whole catalogue is now empty, discard entirely
                if not self.library[catalogue]:
                    # print(f'------Discarding entire catalog {catalogue.capitalize()}')
                    cat_discard_list.append(catalogue)
            for discarded_cat in cat_discard_list:
                del self.library[discarded_cat]
        # Discard glasses that do not match the regular expression glass_match
        # or that do match the regular expression glass_exclude
        cat_discard_list = []
        for catalogue in self.library.keys():
            discard_list = []
            for glass in self.library[catalogue].keys():
                if (not re.match(glass_match, glass, flags=re.IGNORECASE)) or re.match(glass_exclude, glass, flags=re.IGNORECASE):
                    # print(f'Discarding {catalogue.capitalize()} {glass} RE mismatch')
                    discard_list.append(glass)
                # Discard glasses based on status
                if select_status:
                    if (self.library[catalogue][glass]['status'] not in select_status) and (glass not in discard_list):
                        discard_list.append(glass)
            # Ditch the discarded glasses
            for discarded_glass in discard_list:
                del self.library[catalogue][discarded_glass]
            # If the whole catalogue is now empty, discard entirely
            if not self.library[catalogue]:
                # print(f'------Discarding entire catalog {catalogue.capitalize()}')
                cat_discard_list.append(catalogue)
        for discarded_cat in set(cat_discard_list):
            del self.library[discarded_cat]            

        self.pressure_ref = 1.0133e5   ## the dispersion measurement default reference pressure (Schott at least), in Pascals
        self.temp_ref = 20.0           ## the dispersion measurement default reference temperature, in degC
        self.rh_ref = 50.0             ## default relative humidity

        if (sampling_domain == 'wavelength'):
            self.waves = np.linspace(wavemin, wavemax, nwaves)      ## wavelength in nm
            self.wavenumbers = 1000.0 / self.waves               ## wavenumber in um^-1
        elif (sampling_domain == 'wavenumber'):
            sigma_min = 1000.0 / wavemax
            sigma_max = 1000.0 / wavemin
            self.wavenumbers = np.linspace(sigma_min, sigma_max, nwaves) ## wavenumber in um^-1
            self.waves = 1000.0 / self.wavenumbers                    ## wavelength in nm
        return

    ## =========================
    # def __getattr__(self, name):
    #     '''
    #     Redirect the default __getattr__() function so that any attempt to generate a currently nonexisting attribute
    #     will trigger a method to generate that attribute from existing attributes.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the attribute being accessed.
    #     '''

    #     if (name == 'nglasses'):
    #         nglasses = 0
    #         for catalog in self.library:
    #             nglasses += len(self.library[catalog].keys())
    #         return(nglasses)
    #     elif (name == 'catalogs'):
    #         catalogs = self.library.keys()
    #         return(catalogs)
    #     elif (name == 'glasses'):
    #         glasses = []
    #         for catalog in self.library:
    #             glasses.extend(self.library[catalog].keys())
    #         return(glasses)
    #     return
    
    def __str__(self):
        '''
        A simple function to print the names of the catalogs and glasses in the
        library. See also pprint() for a more detailed listing of the library.
        Note the difference in syntax. To call this method use `print(gls_lib)`
        whereas to pretty print use `gls_lib.pprint()`.
        '''
        cat, gls = self.get_all_cat_gls()
        if not gls:
            return 'Library is empty.'
        # Want to print in alphabetical order of catalogs and glasses
        cats, glss = zip(*sorted(zip(cat, gls)))
        glass_count = 0
        previous_cat = ''
        print_str = ''
        for cat, gls in zip(cats, glss):
            if cat != previous_cat:
                print_str += f'\nCatalog : {cat} with {len(self.library[cat])} glasses.\n '
                col_count = 0
            previous_cat = f'{cat}'
            print_str += f'{gls:14s}'
            glass_count += 1
            if col_count == 8:
                col_count = 0
                print_str += '\n '
            else:
                col_count += 1
        print_str += f'\nTotal number of glasses in library is {glass_count}'
        return print_str

    def get_num_glasses(self):
        '''
        Returns the updated total number of glasses in the library.
        '''
        num_gls = 0
        for catalog in self.library.keys():
            num_gls += len(self.library[catalog].keys())
        return num_gls

    def delete_glasses(self, catalog=None, glass=[], or_glass_match='a*', and_glass_match='.*', 
                        parm_range={}):
        '''
        Delete the named glass or glasses from the named catalog or catalogs.
        If no catalog is given, then the glass(es) will be deleted from all 
        catalogs in the library, should there be any duplicate names.

        Regular expressions can be provided as further filters.

        Further filtering can then be performed on glass parameters that are
        presently available for the glasses. Only numerical parameters can be selected for
        selection.
        A range for any number of existing glass numeric parameters can be specified
        using a dict in which the key is the parameter name and the value is a
        list or tuple giving the lower and upper bounds for the parameter.

        Parameters
        ----------
        glass : str or list of str
            A glass name (case sensitive) or list of names to delete from the library.
            If no glass is given, only glasses that match the given regular expressions (`or_glass_match`)
            will be deleted. An exact match is required. This is alternative to providing
            a regular expression such as '^S-BSL7$|^S-FPL55$' to delete these two glasses.
        catalog : str or list of str
            A catalog name (case INsensitive) or list of catalog names from which to delete the glasses.
            If catalog is not given or None, all catalogs will be processed.
        or_glass_match : str
            Regular expression to match as an OR condition. That is, named glasses
            AS WELL AS any glasses matching the expression will be deleted.
            If no glasses are named or specified as None, glasses that match
            both regular expressions will be deleted. Matching is case INsensitive.
            Note that 'N-BK7' would match any glass with this sequence of characters,
            including 'N-BK7A'. To be more specific, force matching at the beginning
            and end of the string e.g. '^N-BK7$' which would delete only N-BK7.
        and_glass_match : str
            Regular expresion to match as an AND condition. That is named glasses
            (or glasses matching the `or_glass_match` regular expression)
            will only be deleted if they ALSO match the provided regular expression
            input `and_glass_match`. Matching is case INsensitive.
        parm_range : dict
            A set of numeric parameter for the glass to select on e.g. 'nd' for refractive index at d-line.
            If not given or empty dict (default), this selection criterion will not be applied.
            The named parameter must be scalar float, that is stored for each glass in the 
            ZemaxGlassLibrary instance. Some example parameters are:
                'nd'  : refractive index at the d-line
                'vd'  : Abbe number relative to the d-line
                'tce' : thermal coefficient of expansion
                'opto_therm_coeff : opto-thermal coefficient, provided it has been computed using 
                    add_opto_thermal_coeff() method.
            The glass is deleted if any one of the parameter range conditions is met.
            i.e. if the named parameter lies in the given range.
            To delete glasses with the parameter outside the a particular range,
            this method must be called twice, once with the lower deletion range
            e.g. (-np.inf, -10.0) and once with the high deletion range.

        '''
        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]
        # Put both string cases into catalogs and glasses
        cat_upper = [cat.upper() for cat in catalogs]
        cat_lower = [cat.lower() for cat in catalogs]
        catalogs = cat_upper + cat_lower
        # Make sure glass is a list, even if a single glass
        if isinstance(glass, list):
            glasses = glass
        else:
            glasses = [glass]
        gls_upper = [gls.upper() for gls in glasses]
        gls_lower = [gls.lower() for gls in glasses]
        glasses = gls_lower + gls_upper
        cat_discard_list = []
        gls_discard_list = []
        for catalog in self.library.keys():
            for glass in self.library[catalog].keys():
                if (catalog in catalogs) and ((glass in glasses) or 
                        re.match(or_glass_match, glass, flags=re.IGNORECASE)):
                    if re.match(and_glass_match, glass, flags=re.IGNORECASE):
                        if not parm_range:  # no parameter ranges to test
                            gls_discard_list.append(glass)
                            cat_discard_list.append(catalog)
                        else:   # run through the conditions
                            delete_it = False
                            for key in parm_range.keys():
                                parm_val = self.library[catalog][glass][key]
                                delete_it = delete_it or (parm_val >= parm_range[key][0] and
                                                          parm_val <= parm_range[key][1])
                            if delete_it:
                                gls_discard_list.append(glass)
                                cat_discard_list.append(catalog)                               
        # Now do the actual deleting, first glasses
        for (gls, cat) in zip(gls_discard_list, cat_discard_list):
            del self.library[cat][gls]
        # If any catalogue is now empty, discard entirely
        for cat in set(cat_discard_list):
            if not self.library[cat]:
                del self.library[cat]

    def select_glasses(self, catalog=None, glass_match='.*', inplace=True):
        '''
        Select glasses within a library on the basis of a regular expression.

        Parameters
        ----------
        catalog : str or list of str
            Selection will be confined to the catalog or catalogs given.
            Defaults to all catalogs in the library.
        glass_match : str
            A regular expression to match glass names against. If the glass
            matches in multiple catalogs provided in the `catalog` input,
            they will all be selected.
        inplace : boolean
            If set True, a copy is returned with the selected glasses.
            Otherwise the library (`self`) is modified in place.

        Returns
        -------
        None if inplace is True, otherwise returns the selection in a new
        ZemaxGlassLibrary.
        '''
        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]
        # Create a deep copy of the library
        gls_lib = copy.deepcopy(self)
        # Delete the glass library data in the copy
        gls_lib.library = {}
        for catalog in catalogs:
            for glass in self.library[catalog].keys():
                if re.match(glass_match, glass):
                    if not catalog in gls_lib.library.keys():
                        gls_lib.library[catalog] = {}  # Create the catalog
                    gls_lib.library[catalog][glass] = self.library[catalog][glass]
        if inplace:
            self = gls_lib
        else:
            return gls_lib

    def select_glasses_usingDataFrame(self, df, df_col_names, glass_match='.*', inplace=True):
        '''
        Select glasses from a library using catalog and glass names taken from the named columns of
        a pandas DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The pandas DataFrame to use for selection of glasses.
        df_col_names : list of 2-tuples of str
            The column name(s) to use for the selection. The list must comprise
            of 2-tuples with the DataFrame column name of the catalog in the first
            element and DataFrame column name of the corresponding  glasses in
            the second element.
        glass_match : str
            A regular expression to match the glass name against. Only glasses
            matching the RE will be selected. Defaults to '.*', which matches
            all strings.
        inplace : boolean
            If set True, a copy is returned with the selected glasses.
            Otherwise the library (`self`) is modified in place.

        Returns
        -------
        None if inplace is True, otherwise returns the selection in a new
        ZemaxGlassLibrary.                      
        '''
        # Compile the list
        cats = []
        glss = []
        for cat_col, gls_col in df_col_names:
            cats.extend(df[cat_col].values)
            glss.extend(df[gls_col].values)  
        gls_lib = copy.deepcopy(self)
        # Delete the glass library data in the copy
        gls_lib.library = {}
        # Copy over glasses from self as found in the dataframe
        # provided they match the 
        for cat, gls in zip(cats, glss):
            if re.match(glass_match, gls):
                if cat not in gls_lib.library.keys():
                    gls_lib.library[cat] = {}
                gls_lib.library[cat][gls] = copy.deepcopy(self.library[cat][gls])                
        if inplace:
            self = gls_lib
        else:
            return gls_lib


    def merge(self, other, inplace=True):
        '''
        Merges two ZemaxGlassLibrary instances. 

        Also note that after a merge, depending on how the libraries have been
        processed, the data in the resulting merged library may not be
        congruent (have all the same fields/properties).

        Parameters
        ----------
        other : ZemaxGlassLibrary
            The glass library to merge into the self instance.
        inplace : boolean
            If True, will alter the first (self) library in place and
            glasses from the `other` library could overwrite those in `self`.
            If False, a new library is returned without altering either
            library. Default True.

        Returns
        -------
        None if inplace=True. If inplace=False, the merged ZemaxGlassLibrary is
        returned.

        '''
        if not inplace:
            merged_lib = copy.deepcopy(self)
        else:
            merged_lib = self
        for catalog in other.library.keys():
            if catalog not in merged_lib.library.keys():
                merged_lib.library[catalog] = {}
            for glass in other.library[catalog].keys():
                merged_lib.library[catalog][glass] = copy.deepcopy(other.library[catalog][glass])
        # TODO : should warn about inconsistencies, such as air refractive index model
        if merged_lib.air_index_function_name != other.air_index_function_name:
            warnings.warn('Merged libraries have different air index reference functions.')
        if not inplace:
            return merged_lib

    ## =========================
    def pprint(self, catalog=None, glass=None):
        '''
        Pretty-print the glass library, or a chosen catalog in it.

        Parameters
        ----------
        catalog : str or list of str, optional
            The name of the catalog(s) within the library to print. If not given, all catalogs will be printed.
        glass : str
            The name of the glass within the library to print.
        '''

        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]
        print_str = ''
        for catalog in self.library:
            if (catalog not in catalogs): continue
            print_str += f'Catalog : {catalog}'
            for glassname in self.library[catalog]:
                if (glass != None) and (glassname != glass.upper()): continue
                glassdict = self.library[catalog][glassname]
                print_str += '  ' + glassname + ':\n'
                print_str += '    nd       = ' + str(glassdict['nd']) + '\n'
                print_str += '    vd       = ' + str(glassdict['vd']) + '\n'
                print_str += '    dispform = ' + str(glassdict['dispform']) + \
                      ', the ' + ZemaxGlassLibrary.dispformulas[glassdict['dispform']-1] + ' dispersion formula.\n'
                if ('tce' in glassdict):  # thermal coefficient of expansion
                    print_str += '    tce      = ' + str(glassdict['tce']) + ' ppm/K\n'
                if ('density' in glassdict):  # density in g/cc ?
                    print_str += '    density  = ' + str(glassdict['density']) + ' g/cc\n'
                if ('dpgf' in glassdict):  # relative partial dispersion
                    print_str += '    dpgf     = ' + str(glassdict['dpgf']) + '\n'
                if ('cd' in glassdict):  # dispersion formula coefficients
                    print_str += '    cd       = ' + str(glassdict['cd']) + '\n'
                if ('td' in glassdict):  # thermal data
                    print_str += '    td       = ' + str(glassdict['td']) + '\n'
                if ('od' in glassdict):  # environmental data
                    print_str += '    od       = ' + str(glassdict['od']) + '\n'
                if ('ld' in glassdict):  # valid range of dispersion relation
                    print_str += '    ld       = ' + str(glassdict['ld']) + ' is the valid spectral range in microns.\n'
                if ('interp_coeffs' in glassdict):  # interpolation coefficients for polynomial fit
                    print_str += '    coeffs   = ' + repr(glassdict['interp_coeffs']) + '\n'
        if not print_str:
            print('Library is empty.')
        else:
            print(print_str)

    def write_agf(self, filename, catalog=None, glasses=None, encoding='latin'):
        """
        Write a Zemax .agf format glass catalog file for the specified catalogs and glasses.

        Parameters
        ----------
        filename : str
            Name of the file, with path, in which to write the glass catalog data in .agf format.
            The extension should be .agf, but this is not enforced.
        catalog : str or list of str
            The names of the catalogs in the instance to include in the .agf file to be created.
        glasses : str or list of str
            The names of the glasses to include in the new .agf file to be written.
            If None, all glasses in the given catalogs will be written.
            If a particular glass name occurs in more than one catalog, both sets of data
            will be written. It is not known how Zemax will handle multiple glasses with the
            same name in an .agf file.

        Returns
        -------
        success : boolean
            Returns True if writing the file included at least one glass. False
            is returned if the writing the file failed or zero glasses were processed.
        """
        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]
        # Make sure glasses is a list, even if a single glass
        if isinstance(glasses, str):
            glasses = [glasses]
        success = False
        with open(filename, 'wt', encoding=encoding) as agf_file:
            for catalog in self.library:
                if (catalog not in catalogs): continue  # Skip the whole catalog
                for glassname in self.library[catalog]:
                    if (glasses is None) or (glassname in glasses):  # Write this glass
                        agf_file.write(self.library[catalog][glassname]['text'])
                        success = True
        return success

    def abbreviate_cat_names(self, abbreviate_len=1):
        '''
        Abbreviate the catalog names in the glass library to a certain number of characters.
        This helps to shorten catalog listings and can save memory when dealing with long
        lists of glasses. If duplicate abbreviations are found, the abbreviated string
        length is increased by 1, until abbreviations are all unique.

        Parameters
        ----------
        abbreviate_len : int
            Number of characters to abbreviate catalog names to. Must be less than 10
            characters.
        
        '''
        if abbreviate_len > 9:
            raise ValueError('Unable to abbreviate catalog names with less than 10 characters.')
        key_changes = {}
        for old_key in self.library.keys():
            if len(old_key) >= abbreviate_len:
                new_key = old_key[0:abbreviate_len].title()
                if new_key in key_changes.values():  # Oops, already there
                    self.abbreviate_cat_names(abbreviate_len=abbreviate_len+1)
                    return
                else:
                    key_changes[old_key] = new_key
        # Now do the actual changes
        for old_key in key_changes:
            self.library[key_changes[old_key]] = self.library.pop(old_key)


    def asDataFrame(self, fields=['nd', 'vd'], catalog=None, glass=None):
        """
        Return selected glass library data as a pandas DataFrame. By default, the catalog and glass name are always 
        returned under the fields 'cat' and 'gls'.

        Parameters
        ----------
        fields : list of str
            List of fields to include in the DataFrame
            Can be one or more of the following :
                'dispform' : form of the dispersion equation. See the Zemax manual.
                'nd'  : refractive index at the d line
                'vd'  : standard abbe dispersion number
                'tce' : thermal coefficient of expansion (ppm/K)
                'density' : density in g/cc
                'dpgf' : catalog relative partial dispersion
                'status' : glass status
                'stat_txt' : glass status as a string 'Standard', 'Preferred, 'Obsolete', 'Special' or 'Melt'
                'meltfreq' : Melt frequency of the glass
                'comment' : string comment found in the catalog file
                'relcost' : relative cost of the glass to N-BK7/S-BSL7
                'cr' : Various environmental ratings 
                'fr' : 
                'sr' :
                'ar' : Acid resistance rating
                'pr' : Phosphate resistance rating
            Other fields that have been added to the glass instance should also work.
            Default is ['nd', 'vd'] i.e. refractive index at d-line (587.5618 nm) and standard abbe dispersion number.
        catalog : str or list of str, optional
            The name of the catalog(s) within the library to process. If not given, all catalogs will be processed.
        glass : str
            The name of the glass within the library to process.            

        Returns
        -------
        glass_df : pandas DataFrame
            DataFrame with the requested glass data.

        """
        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]
        # TODO : Check that the fields exist?
        glass_df = pd.DataFrame(columns=['cat', 'gls'] + fields)
        for catalog in self.library:
            if (catalog not in catalogs): continue
            for glassname in self.library[catalog]:
                if (glass != None) and (glassname != glass.upper()): continue
                # Extract the required data
                glassdict = {field: self.library[catalog][glassname][field] for field in fields}
                glassdict['cat'] = catalog
                glassdict['gls'] = glassname
                glass_df = glass_df.append(glassdict, ignore_index=True)
        return glass_df
                

    ## =============================================================================
    def simplify_schott_catalog(self, cat='schott', zealous=False):
        '''
        Remove redundant, little-used, and unusual glasses from the Schott glass catalog.

        This method should be considered deprecated. Instead, consider using
        delete_glasses(), or using regular expression matching when reading catalogs.

        Parameters
        ----------
        zealous : bool, optional
            Whether to remove the "high transmission" and close-to-redundant glasses.
        '''
        if (cat not in self.library):
            return
        schott_glasses = []

        for glass in self.library[cat]:
            schott_glasses.append(glass)
        ## Remove the "inquiry glasses".
        I_glasses = ['FK3', 'N-SK10', 'N-SK15', 'BAFN6', 'N-BAF3', 'N-LAF3', 'SFL57', 'SFL6', 'SF11', 'N-SF19', 'N-PSK53', 'N-SF64', 'N-SF56', 'LASF35']
        ## Remove the "high-transmission" duplications of regular glasses.
        H_glasses = ['LF5HT', 'BK7HT', 'LLF1HT', 'N-SF57HT', 'SF57HT', 'LF6HT', 'N-SF6HT', 'F14HT', 'LLF6HT', 'SF57HHT', 'F2HT', 'K5HT', 'SF6HT', 'F8HT', 'K7HT']
        ## Remove the "soon-to-be-inquiry" glasses from the Schott catalog.
        N_glasses = ['KZFSN5', 'P-PK53', 'N-LAF36', 'UBK7', 'N-BK7']
        ## Remove the Zinc-sulfide and zinc selenide glasses.
        ZN_glasses = ['CLEARTRAN_OLD', 'ZNS_VIS']
        ## "zealous": remove the "P" glasses specifically designed for hot press molding, and several glasses that are nearly identical to others in the catalog.
        Z_glasses = ['N-F2', 'N-LAF7', 'N-SF1', 'N-SF10', 'N-SF2', 'N-SF4', 'N-SF5', 'N-SF57', 'N-SF6', 'N-ZK7', 'P-LASF50', 'P-LASF51', 'P-SF8', 'P-SK58A', 'P-SK60']
        for glass in schott_glasses:
            remove = (glass in I_glasses) or (glass in H_glasses) or (glass in N_glasses) or (glass in ZN_glasses)
            if zealous:
                remove = remove or (glass in Z_glasses)
            if remove:
                del self.library[cat][glass]
        ## Refresh any existing information in the library.
        if hasattr(self, 'nglasses'):
            self.nglasses = self.get_num_glasses()
        if hasattr(self, 'glasses'):
            glasses = []
            for catalog in self.library:
                glasses.extend(self.library[catalog].keys())
            self.glasses = glasses
        return

    def find_nearest_gls(self, catalog, glass, criteria=['nd'], percent=False):
        '''
        Find the glasses in the library that are nearest to a specified glass.
        A number of criteria can be specified for the search.
        Either the absolute or the percentage RMS difference can be used
        as the sort criterion. 

        Parameters
        ----------
        catalog : str
            Catalog of the specified glass for which to find nearby glasses.
        glass : str
            Name of the specified glass for which to find nearby glasses.
        criteria : list of str
            Criteria on which to base the search for nearby glasses.
            The default is ['nd']. The list elements could include the following
                'nd'  : refractive index at the d line
                'vd'  : standard abbe dispersion number
                'tce' : thermal coefficient of expansion (ppm/K)
                'density' : density in g/cc
                'dpgf' : catalog relative partial dispersion
                'opto_therm_coeff' : opto-thermal coefficient, provided that it has
                    been calculated for all glasses using add_opto-thermal_coeff() method.
                'n_rel' : will calculate the rms difference in catalog refractive index over
                    the spectral range defined for the glass library.
        percent : boolean
            If True, will sort by the rms percentage difference relative to the
            values for the specified reference glass/material.
            If False, sort by the RMS of the absolute difference in the defined
            values. Default is False (RMS absolute difference in criteria).
            If only a single criterion is given, there should be virtually no difference
            in the sort order for percent = True or False.   
      
        Returns
        -------
        nearest_glss : pandas dataframe
            Columns of the dataframe are catalog, glass and root-mean-square
            difference in the specified criteria, as well as the root mean
            percentage difference.
        
        '''
        cats, glss = self.get_all_cat_gls()  # Get lists of all catalogs and glasses
        # Find metric vector (columns) for all glasses (rows)
        criteria_array = np.empty((len(glss), 0), dtype=np.float)
        ref_crit = np.array([])
        for criterion in criteria:
            # Any one criterion could be numtiple columns
            # The loop over glasses could therefore produce
            # an array with glasses down rows and criteria across columns
            criterion_array = np.array([])
            for cat, gls in zip(cats, glss):
                # For each glass, build a row vector of values
                crit_vec = np.array([])
                if criterion in self.library[cat][gls].keys():
                    crit_vec = np.array(self.library[cat][gls][criterion])
                elif criterion == 'n_rel':
                    # Calculate the refractive indices at the stored wavelengths
                    _, _, crit_vec = self.get_indices(wv=self.waves, catalog=cat, glass=gls)
                else:
                    raise ValueError(f"Unknown criterion string '{criterion}'")
                if cat == catalog and gls == glass: # This is the reference
                    ref_crit = np.hstack((ref_crit, crit_vec))
                if criterion_array.size == 0:
                    criterion_array = crit_vec
                else:
                    criterion_array = np.vstack((criterion_array, crit_vec))
            criteria_array = np.column_stack((criteria_array, criterion_array))
        # Find RMS difference between metric vectors
        if len(ref_crit) == 0:
            raise ValueError(f'Reference glass/material {glass} in catalogue {catalog} not found.')
        crit_diff = np.sqrt(((criteria_array - ref_crit)**2.0).mean(axis=1))
        crit_percent_diff = np.sqrt((((criteria_array - ref_crit)/ref_crit)**2.0).mean(axis=1)) * 100.0
        if percent:
            return pd.DataFrame({'cat': cats, 'gls':glss, 'rms_diff': crit_diff, 
                    'rms_percent_diff': crit_percent_diff}).sort_values(by=['rms_percent_diff'])
        else:
            return pd.DataFrame({'cat': cats, 'gls':glss, 'rms_diff': crit_diff, 
                    'rms_percent_diff': crit_percent_diff}).sort_values(by=['rms_diff'])            

    ## =========================
    def get_dispersion(self, glass, catalog, T=None, P=None, save_indices=False):
        '''
        For a given glass, calculate the dispersion curve (refractive index as a function of wavelength in nm).

        If sampling_domain=='wavenumber' then the curve is still returned in wavelength units, but the sampling
        will be uniform in wavenumber and not uniform in wavelength. Note that we need to know both the
        catalog and the glass name, and not just the glass name, because some catalogs share the same glass names.

        If the lens thermal data is included, then thermal variation of the index is incorporated into the output.

        Parameters
        ----------
        glass : str
            The name of the glass we want to know about.
        catalog : str
            The catalog containing the glass.
        T : float, optional
            The temperature of the lens environment, in degC.
        P : float, optional
            The pressure of the lens environment in Pascals, e.g. air at normal conditions. For vacuum set this value to zero.
        save_indices : boolean, optional
            If set True, the calculated refractive index data will be saved to the glass/catalog instance.
            Do *not* do this if you want to be able to plot the temperature dependence of the refractive index later on.
            Default is False.
        Returns
        -------
        indices : ndarray
            A numpy array giving the sampled refractive index curve.
        '''

        if ('indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['indices'])

        if (catalog == None):
            print('Warning: cannot find glass "' + glass + '" in the library! Aborting ...')
            return(None, None)
        if ('waves' in self.library[catalog][glass]) and ('indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['indices'])

        if T is None:
            T = self.temp_ref
        if P is None:
            P = self.pressure_ref

        if (glass.upper() in ('AIR','VACUUM')):
            cd = None
            ld = np.array((np.amin(self.waves), np.amax(self.waves))) / 1000.0
            dispform = 0
        else:
            cd = self.library[catalog][glass]['cd']
            dispform = self.library[catalog][glass]['dispform']
            ld = self.library[catalog][glass]['ld']

        ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld"
        ## and wavemin,wavemax we first convert the former to nm and then, when done
        ## we convert to um.
        if (np.amax(self.waves) < ld[0] * 1000.0) or (np.amin(self.waves) > ld[1] * 1000.0):
            print('wavemin,wavemax=(%f,%f), but ld=(%f,%f)' % (np.amin(self.waves), np.amax(self.waves), ld[0], ld[1]))
            print('Cannot calculate an index in the required spectral range. Aborting ...')
            return(None, None)

        ## Choose which domain is the one in which we sample uniformly. Regardless
        ## of choice, the returned vector "w" gives wavelength in um.
        if (self.sampling_domain == 'wavelength'):
            w = self.waves / 1000.0     ## convert from nm to um
        elif (self.sampling_domain == 'wavenumber'):
            w = self.wavenumbers

        if ('td' in self.library[catalog][glass]):
            td = self.library[catalog][glass]['td']
            T_ref = td[6]       ## the dispersion measurement reference temperature in degC
        else:
            td = np.zeros(6)
            T_ref = 0.0        ## the dispersion measurement reference temperature in degC

        ## Calculating the index of air is a special case, for which we can give a fixed formula.
        ## Watch out for region of wavelength validity. The formula could be Kohlrausch, Ciddor or Edlen
        ## Reference : F. Kohlrausch, Praktische Physik, 1968, Vol 1, page 408
        if (glass.upper() == 'AIR'):
            indices = self.air_index_function(w, T, P, self.rh_ref)
        if (glass.upper() == 'VACUUM'):
            indices = np.ones_like(w)

        if (dispform == 0):
            ## use this for AIR and VACUUM
            pass
        else:
            # Get catalog indices
            indices_cat = zemax_dispersion_formula(w, dispform, cd)

        ## If 'TD' is included in the glass data, then include pressure and temperature dependence of the lens
        ## environment. From Schott's technical report "TIE-19: Temperature Coefficient of the Refractive Index".
        ## The above "indices" data are assumed to be from the reference temperature T_ref. Now we add a small change
        ## delta_n to it due to a change in temperature.
        if ('td' in self.library[catalog][glass]):
            td = self.library[catalog][glass]['td']
            dT = T - T_ref
            indices_cat_air = self.air_index_function(w, T=T_ref, P=self.pressure_ref, rh=self.rh_ref)
            indices_abs = indices_cat * indices_cat_air
            dn_abs = ((indices_abs**2 - 1.0) / (2.0 * indices_abs)) * (td[0] * dT + td[1] * dT**2 + td[2] * dT**3 + ((td[3] * dT + td[4] * dT**2) / (w**2 - td[5]**2)))
            indices_abs = indices_abs + dn_abs
            indices_air = self.air_index_function(w, T, P, rh=self.rh_ref)
            indices = indices_abs / indices_air
        else:
            indices = indices_cat

        ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld" with wavemin and wavemax, we need
        ## to multiply by 1000.
        if (np.amin(self.waves) < ld[0] * 1000.0):
            print('Truncating fitting range since wavemin=%fum, but ld[0]=%fum ...' % (np.amin(self.waves)/1000.0, ld[0]))
            indices[self.waves < ld[0] * 1000.0] = np.NaN
        if (np.amax(self.waves) > ld[1] * 1000.0):
            print('Truncating fitting range since wavemax=%fum, but ld[1]=%fum ...' % (np.amax(self.waves)/1000.0, ld[1]))
            indices[self.waves > ld[1] * 1000.0] = np.NaN

        ## Insert result back into the glass data. Do *not* do this if you want to be able to plot the temperature
        ## dependence of the refractive index.
        if save_indices:
            self.library[catalog][glass]['indices'] = indices

        return(self.waves, indices)

    def get_indices(self, wv=wv_d, catalog=None, glass=None, T=None, P=None, rh=50.0):
        '''
        Get the catalog refractive indices of a glass for a specified set of wavelengths, 
        possibly with temperature and pressure corrections.

        Parameters
        ----------
        wv : list or numpy array of float, optional
            Wavelengths at which to compute the refractive indices in nm or in microns.
            If the wavelengths are below 100.0, units are assumed to be microns.
            If the wavelengths are above 100.0, the units are assumed to be nanometers.
            Zemax uses units of microns when specifiying wavelengths.
            Default is the yellow Helium d line at 587.5618 nm
        glass : str or list of str, optional
            Glass(es) for which to calculate the refractive indices. 
            If not provided, all glasses in the catalog(s) will be assumed. 
        catalog : str or list of str, optional
            Catalogs in which to find the specified glass.
            If not provided, all catalogs will be assumed.
        T : float, optional
            The temperature of the lens environment, in degC. The catalog refractive index is nominally corrected 
            for a temperature different from the catalog standard temperature.
        P : float, optional
            The pressure of the lens environment in Pascals, e.g. air at normal conditions. For vacuum set this value to zero.
            The catalog refractive index is nominally corrected for air pressure different from the catalog standard pressure.
            The Schott standard catalogue pressure is 101330 Pa. One standard atmosphere is 101325 Pa.                      

        Returns
        -------
        glasses : list of strings
            List of glass names in the format 'catalog glass'.
        indices : ndarray of float
            Array of refractive indices. Wavelength varies across columns. Glasses vary down rows.             
        '''
        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]
        catalog_list = []  # This will be compiled per glass and returned
        glass_list = []
        wv = np.asarray(wv, dtype=np.float)
        # Scale wavelengths to microns
        unit_scaling = np.asarray(wv > 100.0, dtype=np.float) / 1000.0 + np.asarray(wv <= 100.0, dtype=np.float)
        wv *= unit_scaling  # Zemax formulae assume microns        

        if T is None:
            T = self.temp_ref
        if P is None:
            P = self.pressure_ref

        indices = np.asarray([])
        for this_catalog in self.library:
            if (this_catalog not in catalogs): continue
            if (glass == None):
                glasses = self.library[this_catalog].keys()
            elif isinstance(glass, list):
                glasses = glass
            else:
                glasses = [glass]
            for this_glass in glasses:
                if this_glass not in self.library[this_catalog].keys(): continue  # Skip the glass if not in this catalog
                catalog_list.append(this_catalog)
                glass_list.append(this_glass)
                # Calculate the refractive index (catalog indices in air at catalog reference temperature and air pressure)
                indices_cat = zemax_dispersion_formula(wv, self.library[this_catalog][this_glass]['dispform'],
                                                             self.library[this_catalog][this_glass]['cd'])
                # Perform a temperature and pressure correction if temperature or pressure is different from the reference and the data is available
                if ('td' in self.library[this_catalog][this_glass]):
                    td = self.library[this_catalog][this_glass]['td']
                    T_ref = td[6]  # Fetch the catalogue reference temperature for this glass usually 20 C
                    dT = T - T_ref  # difference between catalog reference temperature and requested environmental temperature
                    # Compute the index of air for reference catalog conditions
                    indices_cat_air = self.air_index_function(wv, T=T_ref, P=self.pressure_ref, rh=self.rh_ref)
                    # Compute the absolute glass indices (always higher relative to vacuum)
                    indices_abs = indices_cat * indices_cat_air
                    # Compute the change in absolute refractive index
                    dn_abs = ((indices_abs**2 - 1.0) / (2.0 * indices_abs)) * (td[0] * dT + td[1] * dT**2 + td[2] * dT**3 + ((td[3] * dT + td[4] * dT**2) / (wv**2 - td[5]**2)))
                    indices_abs = indices_abs + dn_abs
                    # Compute the environmental air indices
                    indices_env_air = self.air_index_function(wv, T, P, rh)
                    # Finally, compute the glass indices at the requested environmental conditions
                    glass_indices = indices_abs / indices_env_air
                else: # Just return the catalog indices corrected for pressure
                    # Compute indices for air at standard catalog environmental conditions
                    indices_cat_air = self.air_index_function(wv, T=self.pressure_ref, P=self.pressure_ref, rh=self.rh_ref)
                    # Compute air indices for requested environment
                    indices_env_air = self.air_index_function(wv, T, P, rh)
                    # Correct catalog indices for difference in catalog and environmental conditions
                    glass_indices = indices_cat * indices_cat_air / indices_env_air
                if indices.size > 0:
                    indices = np.vstack((indices, glass_indices))
                else:
                    indices = np.atleast_1d(glass_indices)
        
        return catalog_list, glass_list, indices

    def add_opto_thermal_coeff(self, temp_lo, temp_hi, wv_ref=wv_d, pressure_env=101330.0):
        '''
        Compute the opto-thermal coefficients for all glasses in the library for a particular temperature range.
        The glass must have TD temperature data for the opto-thermal coefficient to be valid, as well as a
        valid coefficient of thermal expansion.

        The coefficient is added to the glass data as the dictionary value 'opto_therm_coeff'. Units are per Kelvin.
        The dn/DT (thermal coefficient of refractive index) is also added as the dictionary value 'dndT'.

        Parameters
        ----------
        temp_lo : float
            Low temperature for calculation of opto-thermal coefficents in degrees C.
        temp_hi : float
            High temperature for calculation of opto-thermal coefficients in degrees C.
        wv_ref : float
            Reference (centre) wavelength at which to compute the opto-thermal coefficients. 
            If any wavelength is greater than 100 it is assumed that units are nm.
        pressure_env : float
            Air pressure for calculation of opto-thermal constants.
            Default is 101330 Pa, the Schott catalog reference.
            To calculate absolute opto-thermal coefficents, set pressure_env=0.0 (vacuum).            
        '''
        self.temp_lo = temp_lo
        self.temp_hi = temp_hi
        self.wv_ref = wv_ref
        self.press_env = pressure_env
        # Compute refractive indices at the low temperature
        cat_list, gls_list, ind_lo = self.get_indices(wv_ref, T=temp_lo, P=pressure_env)
        # Compute refractive indices at the reference temperature
        cat_list, gls_list, ind_ref = self.get_indices(wv_ref, P=pressure_env)
        # Compute refractive indices at the high temperature
        cat_list, gls_list, ind_hi = self.get_indices(wv_ref, T=temp_hi, P=pressure_env)
        # Compute an array of dn/dT using low and high temperature indices
        dndT = (ind_hi - ind_lo) / (temp_hi - temp_lo)
        # Compute opto-thermal coefficients and insert into database
        for i_gls in range(len(gls_list)):
            opto_therm_coeff = dndT[i_gls] / (ind_ref[i_gls] - 1.0) - self.library[cat_list[i_gls]][gls_list[i_gls]]['tce']/1.0e6
            self.library[cat_list[i_gls]][gls_list[i_gls]]['opto_therm_coeff'] = float(opto_therm_coeff)
            self.library[cat_list[i_gls]][gls_list[i_gls]]['dndT'] = float(dndT[i_gls])
            self.library[cat_list[i_gls]][gls_list[i_gls]]['n_ref'] = float(ind_ref[i_gls])

    def get_abbe_number(self, wv_centre=wv_d, wv_lo=wv_F, wv_hi=wv_C, catalog=None, glass=None):
        '''
        Calculate generalised Abbe numbers for glasses.

        Parameters
        ----------
        wv_centre : float, optional
            The centre wavelength to use for the generalised Abbe number calculation.
            Default is the yellow Helium d line at 587.5618 nm
        wv_lo : float, optional
            The low wavelength to use for the generalised Abbe number calculation.
            Default is the red Hydrogen C line at 656.2725 nm
        wv_hi : float, optional
            The high wavelength to use for the generalised Abbe number calculation.
            Default is the blue Hydrogen F line at 486.1327 nm
        glass : str or list of str, optional
            Glass(es) for which to calculate the refractive indices. 
            If not provided, all glasses in the catalog(s) will be assumed. 
        catalog : str or list of str, optional
            Catalogs in which to find the specified glass.
            If not provided, all catalogs will be assumed.       

        Returns
        -------
        cat_names : list of str
            List of catalogs for the follownig list of glasses.
        glass_names : list of str
            List of same length as cat_names, giving the glasses.
        abbe_numbers : ndarray of float
            Array of generalised Abbe numbers. Same length as cat_names and glass_names.
        '''
        wv = np.asarray([wv_centre, wv_lo, wv_hi], dtype=np.float)
        # First calculate the refractive indices at the relevant wavelengths
        cat_names, glass_names, indices = self.get_indices(wv, catalog=catalog, glass=glass)
        # Calculate the generalised Abbe Number
        abbe_numbers = (indices[:, 0] - 1.0) / (indices[:, 1] - indices[:, 2])
        return cat_names, glass_names, abbe_numbers

    def get_relative_partial_dispersion(self, wv_x=wv_g, wv_y=wv_F, wv_lo=wv_F, wv_hi=wv_C, catalog=None, glass=None):
        '''
        Calculate generalised relative partial dipersion for glasses.
        The relative partial dispersion is typically the drop in refractive index over a wavelength step in the shorter
        wavelength region divided by the drop in wavelength over a wavelength step in a longer wavelength region.
        The default calculation is for P_g,F relative partial dispersion, which is typically also listed
        explicitly in the catalog.

        Parameters
        ----------
     
        wv_x : float, optional
            The numerator lower wavelength to use for the generalised partial dispersion number calculation.
            Default is the deep blue Mercury g line at 435.8343 nm
        wv_y : float, optional
            The numerator upper wavelength to use for the generalised partial dispersion calculation.
            Default is the blue Hydrogen F line at 486.1327 nm         
        wv_lo : float, optional
            The denominator lower wavelength to use for the generalised partial dispersion calculation.
            Default is the red Hydrogen C line at 656.2725 nm
        wv_hi : float, optional
            The denominator high wavelength to use for the generalised partial dispersion calculation.
            Default is the blue Hydrogen F line at 486.1327 nm
        glass : str or list of str, optional
            Glass(es) for which to calculate the refractive indices. 
            If not provided, all glasses in the catalog(s) will be assumed. 
        catalog : str or list of str, optional
            Catalogs in which to find the specified glass.
            If not provided, all catalogs will be assumed.       

        Returns
        -------
        glasses : list of strings
            List of glass names in the format 'catalog glass'.
        rel_partial_dispersion : ndarray of float
            Array of generalised relative partial dispersions.  
        '''
        wv = np.asarray([wv_x, wv_y, wv_lo, wv_hi], dtype=np.float)
        # First calculate the refractive indices at the relevant wavelengths
        cat_names, glass_names, indices = self.get_indices(wv, catalog=catalog, glass=glass)
        # Calculate the generalised Abbe Number
        rel_partial_dispersion = (indices[:, 0] - indices[:, 1]) / (indices[:, 2] - indices[:, 3])
        return cat_names, glass_names, rel_partial_dispersion

    def get_pair_rank_color_correction(self, wv_centre=wv_d, wv_x=wv_g, wv_y=wv_F, wv_lo=wv_F, wv_hi=wv_C, 
                                    catalog=None, glass=None, as_df=False):
        '''
        Get glass pairwise ranks for color correction potential based on generalised Abbe number and relative partial dispersion.
        The ranking is calculated as the difference in generalised Abbe number divided by the difference in generalised
        relative partial dispersion. Glass pairs are favoured for color correction if there is a small difference in
        relative partial dispersion combined with a large difference in Abbe number.

        Parameters
        ----------
        wv_centre : scalar float or array of float
            The centre wavelength to use for the generalised Abbe number calculation.
            Default is the yellow Helium d line at 587.5618 nm.
            If wv_centre is an array, then all the wavelength inputs must also be arrays of the same length.
            In this case, the color correction merit is the product of the merit values for the different
            wavelength regions effectively so specified.           
        wv_x : scalar float or array of float
            The numerator lower wavelength to use for the generalised partial dispersion calculation.
            Default is the deep blue Mercury g line at 435.8343 nm
        wv_y : scalar float or array of float
            The numerator upper wavelength to use for the generalised partial dispersion calculation.
            Default is the blue Hydrogen F line at 486.1327 nm         
        wv_lo : scalar float or array of float
            The denominator lower wavelength to use for the generalised partial dispersion calculation.
            Default is the red Hydrogen F line at 486.1327 nm
        wv_hi : scalar float or array of float
            The denominator high wavelength to use for the generalised partial dispersion calculation.
            Default is the blue Hydrogen C line at 656.2725 nm
        glass : str or list of str, optional
            Glass(es) for which to calculate the outputs. 
            If not provided, all glasses in the catalog(s) will be assumed. 
        catalog : str or list of str, optional
            Catalogs in which to find the specified glass.
            If not provided, all catalogs will be assumed. 
        as_df : bool
            If set True, the data will be returned as a pandas DataFrame with columns named as in the Returns.
            Default False.
  

        Returns
        -------
        if as_df is False, the following returns are to be expected
        cat1 : list of strings
            Catalog from which glass1 is taken
        cat2 : list of strings
            Catalog from which glass 2 is taken
        gls1 : list of strings
            First glass in pair, ranked by color correction potential
        gls2 : list of strings
            Second glass in pair in same rank order as all other outputs
        merit : ndarray of float
            Array of color correction potential merit. The higher the merit value, the better the potential
            color correction based on the ratio of the generalised relative partial disperion to 
            generalised abbe number.
        
        If as_df is set True, the above are returned as the columns in a pandas DataFrame.
        '''
        wv_centre, wv_x, wv_y = np.atleast_1d(wv_centre), np.atleast_1d(wv_x), np.atleast_1d(wv_y)
        wv_lo, wv_hi = np.atleast_1d(wv_lo), np.atleast_1d(wv_hi)
        if wv_centre.size != wv_x.size or wv_centre.size != wv_y.size or wv_centre.size != wv_lo.size or wv_centre.size != wv_hi.size:
            raise ValueError('Wavelength inputs must all be the same length.')
        merit = None
        for i_wv in range(wv_centre.size):
            # Calculate the generalised Abbe numbers
            cat_names, glass_names, abbe_number = self.get_abbe_number(wv_centre=wv_centre[i_wv], wv_lo=wv_lo[i_wv], wv_hi=wv_hi[i_wv], 
                                                            catalog=catalog, glass=glass)
            # Calculate the generalised relative partial dispersion
            cat_names, glass_names, rel_part_disp = self.get_relative_partial_dispersion(wv_x=wv_x[i_wv], wv_y=wv_y[i_wv], 
                                        wv_lo=wv_lo[i_wv], wv_hi=wv_hi[i_wv], catalog=catalog, glass=glass)
            # Replicate the matrices up to two dimensions
            abbe_number, rel_part_disp = np.meshgrid(abbe_number, rel_part_disp)
            # Calculate the difference in Abbe number divided by the difference in relative partial dispersion
            this_merit = np.abs(abbe_number - abbe_number.T) / np.abs(rel_part_disp - rel_part_disp.T)
            # Set all resulting Nan values to zero
            this_merit = np.nan_to_num(this_merit)
            if merit is None:
                merit = this_merit
            else:
                merit *= this_merit
        cat1, cat2 = np.meshgrid(np.array(cat_names), np.array(cat_names))
        glass1, glass2 = np.meshgrid(np.array(glass_names), np.array(glass_names))
        rank_order = np.flip(merit.flatten().argsort())[::2]  # Take every second one because they are duplicated
        # The last elements corresponding to the diagonal can also be discarded
        rank_order = rank_order[:-merit.shape[0]]
        if as_df:  # Return as a dataframe
            the_data = {'cat1': cat1.flatten()[rank_order], 'gls1': glass1.flatten()[rank_order],
                        'cat2': cat2.flatten()[rank_order], 'gls2': glass2.flatten()[rank_order],
                        'merit': merit.flatten()[rank_order]}
            return pd.DataFrame(data=the_data)
        else:
            return cat1.flatten()[rank_order], cat2.flatten()[rank_order], \
               glass1.flatten()[rank_order], glass2.flatten()[rank_order], merit.flatten()[rank_order]
    
    def get_pair_rank_index_ratio(self, wv_lo=wv_F, wv_hi=wv_C, n_wv=5,
                                catalog=None, glass=None, as_df=False):
        '''
        Get glass pairwise ranks for color correction potential based on refractive index ratio.
        The ranking is calculated as the variance of the refractive index ratio computed over a specified
        wavelength range.

        Parameters
        ----------
        wv_lo : scalar float or array of float
            The lower wavelength limit for computation of the index ratio variance merit.
            Default is the red Hydrogen F line at 486.1327 nm
        wv_hi : scalar float or array of float
            The upper wavelength limit for computation of the index ratio variance merit.
            Default is the blue Hydrogen C line at 656.2725 nm
        glass : str or list of str, optional
            Glass(es) for which to calculate the outputs. 
            If not provided, all glasses in the catalog(s) will be assumed. 
        catalog : str or list of str, optional
            Catalogs in which to find the specified glass.
            If not provided, all catalogs will be assumed. 
        as_df : bool
            If set True, the data will be returned as a pandas DataFrame with columns named as in the Returns.
            Default False.
  
        Returns
        -------
        if as_df is False, the following returns are to be expected
        cat1 : list of strings
            Catalog from which glass1 is taken
        cat2 : list of strings
            Catalog from which glass 2 is taken
        gls1 : list of strings
            First glass in pair, ranked by color correction potential
        gls2 : list of strings
            Second glass in pair in same rank order as all other outputs
        merit : ndarray of float
            Array of color correction potential merit. The higher the merit value, the better the potential
            color correction based on index ratio variance.
        
        If as_df is set True, the above are returned as the columns in a pandas DataFrame.
        '''
        wv_lo, wv_hi = np.atleast_1d(wv_lo), np.atleast_1d(wv_hi)
        if wv_lo.size != wv_hi.size:
            raise ValueError('Wavelength inputs must all be the same length.')
        merit = None
        for i_wv in range(wv_lo.size):
            wv = np.linspace(wv_lo[i_wv], wv_hi[i_wv], n_wv)
            # Calculate the indices for all the glasses
            cat_names, glass_names, indices = self.get_indices(wv=wv, catalog=catalog, glass=glass)
            # Run through all the glasses and compute ratio variance compared to all other glasses
            n_glss = len(glass_names)  # Number of glasses
            ratio_stdev = np.zeros((n_glss, n_glss))
            # Calculate the ratios only for distinct glass pairs, the diagonal will be zeros
            for i_gls in range(n_glss):
                for j_gls in range(n_glss):
                    if i_gls != j_gls:
                        ratio_stdev[i_gls, j_gls] = np.std(indices[i_gls, :] / indices[j_gls, :])
            this_merit = 1.0 / ratio_stdev**2.0  # Zeros will become inf
            this_merit[np.isinf(this_merit)] = 0.0
            if merit is None:
                merit = this_merit
            else:
                merit *= this_merit
        cat1, cat2 = np.meshgrid(np.array(cat_names), np.array(cat_names))
        glass1, glass2 = np.meshgrid(np.array(glass_names), np.array(glass_names))
        rank_order = np.flip(merit.flatten().argsort())
        # The last elements corresponding to the diagonal can also be discarded
        rank_order = rank_order[:-merit.shape[0]]
        if as_df:  # Return as a dataframe
            the_data = {'cat1': cat1.flatten()[rank_order], 'gls1': glass1.flatten()[rank_order],
                        'cat2': cat2.flatten()[rank_order], 'gls2': glass2.flatten()[rank_order],
                        'merit': merit.flatten()[rank_order]}
            return pd.DataFrame(data=the_data)
        else:
            return cat1.flatten()[rank_order], cat2.flatten()[rank_order], \
               glass1.flatten()[rank_order], glass2.flatten()[rank_order], merit.flatten()[rank_order]
    
    def supplement_df(self, pd_df, fields):
        """
        Supplement a pandas dataframe with data from the glass catalog.
        This method will return the pandas dataframe with additional data taken from the glass catalog.

        Parameters
        ----------
        pd_df : pandas dataframe
            Dataframe to be supplemented. Must have at least one column starting with 'cat' and another, corresponding
            column starting with 'gls'.
        fields : list of str
            List of field names to extracted from the glass catalog data. See method asDataFrame() for some examples.

        Returns
        -------
        pd_df : pandas dataframe
            Contains input dataframe extended with requested data from the glass library instance (self).
        """
        # Get the columns
        columns = pd_df.keys()
        for column in columns:
            if column.startswith('cat'):
                cat_col = column
                gls_col = 'gls' + cat_col[3:]
                cats = pd_df[cat_col]
                glss = pd_df[gls_col]
                for field in fields:
                    # Fetch the requested data from the library
                    col_dat = []
                    for i_gls in range(len(glss)):
                        col_dat.append(self.library[cats[i_gls]][glss[i_gls]][field])
                    # Add the new column
                    new_col = field + gls_col[3:]
                    if new_col in columns:
                        # Delete column
                        pd_df = pd_df.drop(columns=new_col)
                    #print('Length '+new_col+'  '+str(len(col_dat)))
                    #print(col_dat)
                    pd_df.insert(loc=len(pd_df.columns), column=new_col, value=col_dat)
        return pd_df

    def get_all_cat_gls(self):
        '''
        Get pair-wise (same length) lists of all the catalogs and glasses in
        the library.

        Returns
        -------
        cat : list of str
            Catalog names for all glasses in the library.
        gls : list of str
            All glass names in the library (same length as cat).
        '''
        cat = []
        gls = []
        for catalog in self.library.keys():
            cat.extend([catalog] * len(self.library[catalog].keys()))
            gls.extend(self.library[catalog].keys()) 
        return cat, gls   

    ## =========================
    def get_polyfit_dispersion(self, glass, catalog):
        '''
        Get the polynomial-fitted dispersion curve for a glass.

        Note that we need to know both the catalog and the glass name, and not just the glass name,
        because some catalogs share the same glass names.

        Parameters
        ----------
        glass : str
            Which glass to analyze.
        catalog : str
            The catalog containing the glass.
        '''

        if ('interp_indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['interp_indices'])

        ## Generate a vector of wavelengths in nm, with samples every 1 nm.
        (waves, indices) = self.get_dispersion(glass, catalog)

        okay = (indices > 0.0)
        if not any(okay):
            return(waves, np.ones_like(waves) * np.NaN)

        x = np.linspace(-1.0, 1.0, np.alen(waves[okay]))
        coeffs = np.polyfit(x, indices[okay], self.degree)
        coeffs = coeffs[::-1]       ## reverse the vector so that the zeroth degree coeff goes first
        self.library[catalog][glass]['interp_coeffs'] = coeffs

        interp_indices = polyeval_Horner(x, coeffs)
        self.library[catalog][glass]['interp_indices'] = interp_indices

        return(waves, interp_indices)

    def buchdahl_find_alpha(self, wv, wv_center, order=3, catalog=None, glass=None, gtol=1.0e-9, 
                            show_progress=False):
        """
        Fit a Buchdahl dispersion function to the specified glasses and catalogs.
        Every glass will return with its own alpha value and therefore the listed Buchdahl
        coefficients are NOT COMPATIBLE with each other for glass selection purposes.

        Once the optimal alpha value for a collection of glasses has been selected using this
        method, the method buchdahl_fit() should be used with the selected alpha, which
        will then recompute the Buchdahl fit using a consistent alpha for all glasses in the
        collection.


        Parameters
        ----------
        wv : array of float
            Wavelength samples to use for performing the fit. If > 100.0, units of nm are assumed, otherwise microns.
            Must be in increasing order.
        wv_center : float
            The center (reference) wavelength to use for the Buchdahl fit. Should be in the range of wv.
        order : int, optional
            The order of the Buchdahl fit. Only order 2 and 3 are supported. Defaults to 3.
        catalog : list of str, optional
            A list of catalogs to be processed. Defaults to all catalogs in the library.
        glass : list of string, optional
            List of glasses to process. Defaults to all glasses in the catalogs.
        gtol : float, optional
            Convergence demand parameter for best fitting of Buchdahl alpha so as to provide as close to linear
            relationship between Buchdahl omega spectral variable and refractive index.
            Defaults to 1.0e-9, but 1.0e-8 can be considered to speed up computational convergence.
        show_progress : boolean
            If set True, shows a simple text progress bar for all the requested glasses.
            Default is False (no progress bar is shown). Only works if IPython is installed.

        Returns
        -------
        cat_list : list of str
            List of catalogs from which the corresponding glasses come.
        glass_list : list of str
            List of glass names, same length as cat_list.
        buch_fits : ndarray of float
            Buchdahl fit parameters for the listed glasses, where buch_fits[:, 0] is alpha,
            buch_fits[:, 1] is nu_1, buch_fits[:, 2] is nu_2 and (if order==3) buch_fits[:, 3] is nu_3 


        """
        # Determine the refractive indices of the glasses at the given wavelengths
        cat_list, glass_list, indices = self.get_indices(wv, catalog=catalog, glass=glass)
        # Determine the refractive index of the glass at the center wavelength
        cat_list, glass_list, n_centers = self.get_indices(wv_center, catalog=catalog, glass=glass)
        # Find the most linear alpha value and Buchdahl coefficients (2 or 3)
        for i_glass in range(len(glass_list)):
            # The following will return a list of float, alpha, nu_1, nu_2 and (if order==3) nu_3
            buch_fit = buchdahl_find_alpha(wv, indices[i_glass, :], wv_center, n_centers[i_glass], order=order, gtol=gtol)
            if i_glass == 0:
                buch_fits = np.array(buch_fit['x'])
            else:
                buch_fits = np.vstack((buch_fits, np.array(buch_fit['x'])))
            if show_progress and clear_output_possible:
                update_progress((i_glass + 1.0) / len(glass_list), bar_length=50)
        return cat_list, glass_list, buch_fits


    def buchdahl_fit(self, wv, wv_center, alpha, order=3, catalog=None, glass=None, 
                            show_progress=False):
        """
        Fit a Buchdahl dispersion function to the specified glasses and catalogs.
        An appropriate Buchdahl alpha parameter value must be provided.
        This can be determined for the glass catalog using the buchdahl_find_alpha() method.


        Parameters
        ----------
        wv : array of float
            Wavelength samples to use for performing the fit. If > 100.0, units of nm are assumed, otherwise microns.
            Must be in increasing order.
        wv_center : float
            The center (reference) wavelength to use for the Buchdahl fit. Should be in the range of wv.
        alpha : float
            The Buchdahl alpha parameter to use for fitting. Can be determined using buchdahl_find_alpha().
            Alternatively, give a generic value of about 2.5 for typical optical glass catalogs in the visible spectrum.
        order : int, optional
            The order of the Buchdahl fit. Only order 2 and 3 are supported. Defaults to 3.
        catalog : list of str, optional
            A list of catalogs to be processed. Defaults to all catalogs in the library.
        glass : list of string, optional
            List of glasses to process. Defaults to all glasses in the catalogs.
        show_progress : boolean
            If set True, shows a simple text progress bar for all the requested glasses.
            Default is False (no progress bar is shown). Only works if IPython is installed.
            Useful for interactive work in notebooks.

        Returns
        -------
        cat_list : list of str
            List of catalogs from which the corresponding glasses come.
        glass_list : list of str
            List of glass names, same length as cat_list.
        buch_fits : ndarray of float
            Buchdahl fit parameters (nu) for the listed glasses, where buch_fits[:, 0] is alpha,
            buch_fits[:, 1] is nu_1, buch_fits[:, 2] is nu_2 and (if order==3) buch_fits[:, 3] is nu_3
        n_centers : ndarray of float
            Refractive index at the center (reference) wavelength.


        """
        # Determine the refractive indices of the glasses at the given wavelengths
        cat_list, glass_list, indices = self.get_indices(wv, catalog=catalog, glass=glass)
        # Determine the refractive index of the glass at the center wavelength
        cat_list, glass_list, n_centers = self.get_indices(wv_center, catalog=catalog, glass=glass)
        # Find the Buchdahl coefficients (2 or 3), with the given alpha value.
        for i_glass in range(len(glass_list)):
            # The following will return a list of float, alpha, nu_1, nu_2 and (if order==3) nu_3
            buch_fit = buchdahl_fit(wv, indices[i_glass, :], wv_center, n_centers[i_glass], alpha=alpha, order=order)

            if i_glass == 0:
                buch_fits = buch_fit
            else:
                buch_fits = np.vstack((buch_fits, buch_fit))
            if show_progress and clear_output_possible:
                update_progress((i_glass + 1.0) / len(glass_list), bar_length=50)
        return cat_list, glass_list, buch_fits, n_centers

    def buchdahl_fit_eta(self, wv, i_wv_ref, alpha, catalog=None, glass=None, 
                            show_progress=False):
        """
        Find eta coefficients of Buchdahl dispersive power function for the specified glasses and catalogs.
        An appropriate Buchdahl alpha parameter value must be provided.
        The best value of the Buchdahl alpha parameter can be determined for the glass catalog 
        using the buchdahl_find_alpha() method.

        The reference wavelength must be one of the given wavelengths.

        The polynomial order of the fit will be one less than the number of wavelengths (n=len(wv)), but
        any number of wavelengths can be provided resulting in n-1 eta coefficients for each glass.

        This method uses linear algebra to find an exact (within numerical precision) solution.

        Parameters
        ----------
        wv : array of float
            Wavelength samples to use for performing the fit. If > 100.0, units of nm are assumed, otherwise microns.
            Must be in increasing order and distinct.
        i_wv_ref : int
            The index of the reference wavelength to use for the Buchdahl fit. Must be within the length of the wv input. 
        alpha : float
            The Buchdahl alpha parameter to use for fitting. Can be determined using buchdahl_find_alpha().
            Alternatively, give a generic value of about 2.5 for typical optical glass catalogs in the visible spectrum.
        catalog : list of str, optional
            A list of catalogs to be processed. Defaults to all catalogs in the library.
        glass : list of string, optional
            List of glasses to process. Defaults to all glasses in the catalogs.
        show_progress : boolean
            If set True, shows a simple text progress bar for all the requested glasses.
            Default is False (no progress bar is shown). Only works if IPython is installed.
            Useful for interactive work in notebooks.

        Returns
        -------
        cat_list : list of str
            List of catalogs from which the corresponding glasses come.
        glass_list : list of str
            List of glass names, same length as cat_list.
        buch_etas : ndarray of float
            Buchdahl dispersive power fit parameters (eta) for the listed glasses.
            The number of coefficents is one less than the number of wavelengths
            provided (wv). One row of coefficients per glass.
        n_ref : ndarray of float
            Refractive index at the reference wavelength, input wv[i_wv_ref].
        
        """
        n_wv = len(wv)  # The number of wavelengths
        # Determine the catalog refractive indices of the glasses at the given wavelengths
        cat_list, glass_list, indices = self.get_indices(wv, catalog=catalog, glass=glass)
        n_ref = indices[:, i_wv_ref]  # Extract the indices at the reference wavelength
        # Compute the Buchdahl omega values at the wavelengths
        omega = buchdahl_omega(wv, wv[i_wv_ref], alpha)[:, np.newaxis]
        # Form a matrix of n_wv rows by n_wv-1 columns, where power of omega increase across columns
        # There is a row for each omega value so the first column is just the omega values
        omega_mat = omega  # First column is just omega**1
        for i_wv in range(2, n_wv):  # Add higher powers of omega
            omega_mat = np.hstack((omega_mat, omega**i_wv))
        # Delete the row of zeros corresponding to the reference wavelength to get a square matrix
        omega_mat = np.delete(omega_mat, i_wv_ref, axis=0)
        # Also delete the reference refractive index from the indices matrix
        indices = np.delete(indices, i_wv_ref, axis=1)
        # Invert the omega_mat matrix for later use
        omega_mat_inv = np.linalg.inv(omega_mat)
        for i_glass in range(len(glass_list)):    
            # Compute the dispersive power at all wavelengths for this glass
            # Dispersive power is by definition also zero at the reference wavelength
            dispersive_power = (indices[i_glass, :] - n_ref[i_glass]) / (n_ref[i_glass] - 1.0)
            buch_eta = np.matmul(omega_mat_inv, dispersive_power)
            if i_glass == 0:
                buch_etas = buch_eta  # Start
            else:
                buch_etas = np.vstack((buch_etas, buch_eta))  # Add result for this glass
            if show_progress and clear_output_possible:
                update_progress((i_glass + 1.0) / len(glass_list), bar_length=50)
        return cat_list, glass_list, buch_etas, n_ref               


    ## =============================================================================
    def cull_library(self, key1, tol1, key2=None, tol2=None):
        '''
        Reduce all catalogs in the library such that no two glasses are simultaneously
        within (+/- tol1) of key1 and (+/- tol2) of key2.

        Parameters
        ----------
        key1 : str
            The first parameter to analyze. This can be, e.g., "nd" or "dispform". Any key in the \
            glass data dictionary.
        tol1 : float
            The `tolerance` value: if the `key1` properties of any two glasses are within +/-tol1 \
            of one another, then remove all but one from the library.
        key2 : str
            The second parameter to analyze.
        tol2 : float
            The second `tolerance` value: if the `key1` and `key2` properties of any two glasses \
            are within +/-tol1 and +/-tol2 of one another simultaneously, then remove all but one \
            such glass from the library.
        '''

        keydict1 = {}
        keydict2 = {}
        names = []
        keyval1 = []
        keyval2 = []

        for catalog in self.library:
            for glass in self.library[catalog]:
                names.append(catalog + '_' + glass)
                catalogs.append(catalog)

                if (key1 in self.library[catalog][glass]):
                    keyval1.append(self.library[catalog][glass][key1])
                else:
                    keyval1.append(self.library[catalog][glass][None])

                if (key2 != None):
                    if (key2 in self.library[catalog][glass]):
                        keyval2.append(self.library[catalog][glass][key2])
                    else:
                        keyval2.append(self.library[catalog][glass][None])

        names_to_remove = []
        keyval1 = np.array(keyval1)
        keyval2 = np.array(keyval2)

        for i in np.arange(np.alen(names)):
            if (key2 == None):
                idx = np.where(abs(keyval1[i] - keyval1) < tol1)
                names_to_remove.append([name for name in names[idx] if name != names[i]])
            else:
                idx = np.where((abs(keyval1[i] - keyval1) < tol1) and (abs(keyval2 - keyval2[i]) < tol2))
                #print('%3i %3i %5.3f %5.3f %6.3f %6.3f %12s %12s --> REMOVE %3i %12s' % (i, j, keyval1[i], keyval1[j], keyval2[i], keyval2[j], names_all[i], names_all[j], j, names_all[j]))
                names_to_remove.append([name for name in names[idx] if name != names[i]])

        ## Remove the duplicates from the "remove" list, and then delete those glasses
        ## from the glass catalog.
        names_to_remove = np.unique(names_to_remove)
        for glass in names_to_remove:
            (catalog, glass) = glass.split('_')
            #print('i='+str(i)+': catalog='+catalog+'; glass='+name)
            del self.library[catalog][glass]

        return

    ## =========================
    def plot_dispersion(self, glass, catalog, polyfit=False, fiterror=False):
        '''
        Plot the glass refractive index curve as a function of wavelength.

        Parameters
        ----------
        glass : str
            The name of the glass to analyze.
        catalog : str
            The catalog containing the glass.
        polyfit : bool
            Whether to also display the polynomial fit to the curve.
        fiterror : bool
            If `polyfit` is True, then `fiterror` indicates whether a fitting error should also be \
            displayed, using the LHS y-axis.
        '''

        (x, y) = self.get_dispersion(glass, catalog)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'b-', linewidth=2)

        if polyfit:
            (x2, y2) = self.get_polyfit_dispersion(glass, catalog)
            ax.plot(x2, y2, 'ko', markersize=4, zorder=0)

        plt.title(glass + ' dispersion')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('refractive index')

        if polyfit and fiterror:
            fig.subplots_adjust(right=0.85)
            F = plt.gcf()
            (xsize, ysize) = F.get_size_inches()
            fig.set_size_inches(xsize+5.0, ysize)
            err = y2 - y
            ax2 = ax.twinx()
            ax2.set_ylabel('fit error')
            ax2.plot(x2, err, 'r-')

        ## Enforce the plotting range.
        xmin = min(x)
        xmax = max(x)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)

        ymin = min(y)
        ymax = max(y)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)

        ax.axis([xbot,xtop,ybot,ytop])

        return

    def plot_dispersion_diff(self, glass1, catalog1, glass2, catalog2, subtract_mean=False):
        '''
        Plot the difference in refractive index between two glasses (index of second glass minus index of first)

        Parameters
        ----------
        glass1 : str
            The name of the first glass in the difference plot.
        catalog1 : str
            The catalog containing the first glass.
        glass2 : str
            The name of the second glass in the difference plot.
        catalog2 : str
            The catalog containing the second glass.
        subtract_mean : boolean
            If set True, will subtract the mean difference before plotting. Default False.
        '''

        (x1, y1) = self.get_dispersion(glass1, catalog1)
        (x2, y2) = self.get_dispersion(glass2, catalog2)
        rin_diff = y2 - y1
        if subtract_mean:
            rin_diff -= rin_diff.mean()
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(x1, rin_diff, 'b-', linewidth=2)

        plt.title(f'Index Difference between {catalog1.capitalize()} {glass1} and {catalog2.capitalize()} {glass2}')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('refractive index difference')

        ## Enforce the plotting range.
        xmin = min(x1)
        xmax = max(x1)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)

        ymin = min(rin_diff)
        ymax = max(rin_diff)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)

        ax.axis([xbot,xtop,ybot,ytop])

        return

    def plot_dispersion_ratio(self, glass1, catalog1, glass2, catalog2, subtract_mean=False):
        '''
        Plot the ratio in refractive index between two glasses (index of second glass divided by index of first)

        Parameters
        ----------
        glass1 : str
            The name of the first glass in the ratio plot.
        catalog1 : str
            The catalog containing the first glass.
        glass2 : str
            The name of the second glass in the ratio plot.
        catalog2 : str
            The catalog containing the second glass.
        subtract_mean : boolean
            If set True, will subtract the mean ratio before plotting. Default False.
        '''

        (x1, y1) = self.get_dispersion(glass1, catalog1)
        (x2, y2) = self.get_dispersion(glass2, catalog2)
        rin_ratio = y2 / y1
        if subtract_mean:
            rin_ratio -= rin_ratio.mean()
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(x1, rin_ratio, 'b-', linewidth=2)

        plt.title(f'Index Ratio between {catalog1.capitalize()} {glass1} and {catalog2.capitalize()} {glass2}')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('refractive index Ratio')

        ## Enforce the plotting range.
        xmin = min(x1)
        xmax = max(x1)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)

        ymin = min(rin_ratio)
        ymax = max(rin_ratio)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)

        ax.axis([xbot,xtop,ybot,ytop])

        return


    ## =========================
    def plot_temperature_dependence(self, glass, catalog, wavelength_nm, temperatures):
        '''
        Plot the glass refractive index curve as a function of temperature for given wavelengths.

        Parameters
        ----------
        glass : str
            The name of the glass to analyze.
        catalog : str
            The catalog containing the glass.
        wavelength_nm : float
            The wavelength at which to evaluate the temperature dependence.
        temperatures : ndarray
            Array containing the values for which the refractive index shall be plotted.
        '''

        index_vs_temp = []
        for temp in temperatures:
            (waves, indices) = self.get_dispersion(glass, catalog, T=temp)
            res = interp1d(waves*1000.0, indices, wavelength_nm)
            index_vs_temp.append(res)

        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot(111)
        ax.plot(temperatures, index_vs_temp, lw=2)
        plt.title(glass + ' temperature dependence (at %f nm)' % wavelength_nm)
        plt.xlabel('temperatures (degC)')
        plt.ylabel('refractive index')

        return

    ## =========================
    def plot_catalog_property_diagram(self, catalog='all', prop1='nd', prop2='vd', show_labels=True, figure_size=(12, 6)):
        '''
        Plot a scatter diagram of one glass property against another.

        A "property" can be: nd, vd, cr, fr, ar, sr, pr, n0, n1, n2, n3, tce, density, dpgf. Note that
        if "prop1" and "prop2" are left unspecified, then the result is an Abbe diagram.

        If catalog=='all', then all glasses from the entire library are plotted.

        Parameters
        ----------
        catalog : str
            Which catalog to plot.
        prop1 : str
            The glass data property to show along the abscissa (x-axis).
        prop2 : str
            The glass data property to show along the ordinate (y-axis).
        show_labels : bool
            Whether to show the glass name labels near the data points.
        figure_size : tuple of 2 numbers
            Size of plot in x and y, Default is (12, 6)
        '''

        if (catalog == 'all'):
            catalogs = self.library.keys()
        elif isinstance(catalog, list) and (len(catalog) > 1):
            catalogs = catalog
        elif isinstance(catalog, str):
            catalogs = [catalog]

        colors = get_colors(len(catalogs))
        glassnames = []
        all_p1 = []
        all_p2 = []

        fig = plt.figure(figsize=figure_size)
        ax = plt.gca()
        ax.set_prop_cycle(cycler('color', colors))

        ## Collect lists of the property values for "prop1" and "prop2", one catalog at a time.
        ## Plot each catalog separately, so that each can be displayed with unique colors.
        for i,cat in enumerate(catalogs):
            p1 = []
            p2 = []
            for glass in self.library[cat]:
                if (catalog == 'all') and (glass == 'AIR'): continue
                if (catalog == 'all') and (abs(self.library[cat][glass]['vd']) < 1.0E-6): continue

                if (prop1 in ('n0','n1','n2','n3','n4','n5','n6','n7','n8','n9')):
                    j = int(prop1[1])
                    idx = int(prop1[1])
                    if ('interp_coeffs' not in self.library[cat][glass]):
                        #print('Calculating dispersion coefficients for "' + glass + '" ...')
                        self.get_polyfit_dispersion(glass, cat)
                        self.library[cat][glass][prop1] = self.library[cat][glass]['interp_coeffs'][j]
                        #print(glass, self.library[cat][glass]['interp_coeffs'])
                        p2_coeffs = self.library[cat][glass]['interp_coeffs'][j]
                    if ('interp_coeffs' in self.library[cat][glass]):
                        p1_coeffs = self.library[cat][glass]['interp_coeffs'][idx]
                        self.library[cat][glass][prop1] = self.library[cat][glass]['interp_coeffs'][j]
                        p2_coeffs = self.library[cat][glass]['interp_coeffs'][j]
                    else:
                        print('Could not find valid interpolation coefficients for "' + glass + '" glass ...')
                        continue
                else:
                    p2_coeffs = self.library[cat][glass]['interp_coeffs']

                if (prop2 in ('n0','n1','n2','n3','n4','n5','n6','n7','n8','n9')):
                    idx = int(prop2[1])
                    if ('interp_coeffs' not in self.library[cat][glass]):
                        #print('Calculating dispersion coefficients for "' + glass + '" ...')
                        self.get_polyfit_dispersion(glass, cat)
                        p2_coeffs = self.library[cat][glass]['interp_coeffs'][idx]
                    if ('interp_coeffs' in self.library[cat][glass]):
                        p2_coeffs = self.library[cat][glass]['interp_coeffs'][idx]
                    else:
                        print('Could not find valid interpolation coefficients for "' + glass + '" glass ...')
                        continue
                else:
                    p2_coeffs = self.library[cat][glass]['interp_coeffs']

                glassnames.append(glass)

                if (prop1 in ('n0','n1','n2','n3','n4','n5','n6','n7','n8','n9')):
                    p1.append(p1_coeffs)
                if (prop2 in ('n0','n1','n2','n3','n4','n5','n6','n7','n8','n9')):
                    p2.append(p2_coeffs)

            plt.plot(p1, p2, 'o', markersize=5)
            all_p1.extend(p1)
            all_p2.extend(p2)

        plt.title('catalog "' + catalog + '": ' + prop1 + ' vs. ' + prop2)
        plt.xlabel(prop1)
        plt.ylabel(prop2)

        ## Enforce the plotting range.
        xmin = min(all_p1)
        xmax = max(all_p1)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)
        xdist = 0.01 * xrange               ## for plotting text near the data points

        ymin = min(all_p2)
        ymax = max(all_p2)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)
        ydist = 0.01 * yrange               ## for plotting text near the data points

        plt.axis([xbot,xtop,ybot,ytop])
        leg = plt.legend(catalogs, prop={'size':10}, loc='best')
        leg.set_draggable(True)
        #leg = plt.legend(catalogs, prop={'size':10}, bbox_to_anchor=(1.2,1))

        if show_labels:
            ## Plot all of the glass labels offset by (5,5) pixels in (x,y) from the data point.
            trans_offset = offset_copy(ax.transData, fig=fig, x=5, y=5, units='dots')
            for i in np.arange(np.alen(glassnames)):
                #print('i=%i: glassname=%s, p1=%f, p2=%f' % (i, glassnames[i], p1[i], p2[i]))
                plt.text(all_p1[i], all_p2[i], glassnames[i], fontsize=7, zorder=0, transform=trans_offset, color='0.5')

        return


## =============================================================================
## End of ZemaxGlassLibrary class
## =============================================================================

## =============================================================================
## GlassCombo class
## =============================================================================

# First define some combinatorial utility functions
def combinations(n, r):
    '''
    Calculate the number of combinations of n objects taken r at a time
    with no repeats.

    Parameters
    ----------
    n : int
        Total number of items.
    r : int
        Number to choose without repeats.
    '''
    return np.math.factorial(n) // (np.math.factorial(r) * np.math.factorial(n - r))

def split_combos_m_ways(n, r, m):
    '''
    Returns a list of sub-range tuples for n combination r items split into
    m sub-ranges. These ranges are the lexicographical indices, not the 
    combinations themselves. Returns a list of 2-tuples of int.

    Parameters
    ----------
    n : int
        Number of items (glasses in library)
    r : int
        Number of items to choose from n
    m : int
        Number of ranges into which to split the combination indices

    Returns
    -------
    splits : list of int
        Number of combination indices in each split.
    ranges : list of 2-tuples of int
        Index ranges for each of the splits
    
    '''
    total_combinations = combinations(n, r)
    comb_per_sub, remainder = divmod(total_combinations, m)
    splits = [comb_per_sub] * m
    for i in range(remainder):
        splits[i] += 1
    while splits[-1] == 0:  # Ditch trailing zeros
        splits.pop(-1)
    start = []
    stop = []
    pointer = 0
    for split in splits:
        start.append(pointer)
        pointer += split
        stop.append(pointer-1)
    return splits, list(zip(start, stop))

def nth_combination(iterable, r, index):
    '''
    Returns the ith combination taken r at a time of the given iterable.
    Lexicographic order.

    Parameters
    ----------
    iterable : any iterable object
        e.g. range(10) to provide the numbers 0 to 9.
    r : int
        Number to take in each combination
    index : int
        Lexicographic index of the combination to to return.

    Returns
    -------
    combination : tuple of items from the iterable input
        Length of the tuple is equal to r.
    '''
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError
    c = 1
    k = min(r, n-r)
    for i in range(1, k+1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    while r:
        c, n, r = c*r//n, n-1, r-1
        while index >= c:
            index -= c
            c, n = c*(n-r)//n, n-1
        result.append(pool[-1-n])
    return tuple(result)

def gen_combination(start_index, end_index, iterable, r):
    '''
    Given a starting and ending index for a combination of items taken r
    at a time, yields the next combination in lexicographical order.
    Used in conjunction with split_combos_m_ways() to distribute
    nearly equal processing of glass combinations to multiple 
    processors. This is a generator that calls nth_combination().

    Parameters
    ----------
    start_index : int
        Starting index of the combination to yield (inclusive).
    end_index : int
        Ending index (inclusive) of the combination to yield.
    '''
    index = start_index
    while index <= end_index:
        yield nth_combination(iterable, r, index)
        index += 1

def lazy_product(*iter_funcs, **kwargs):
    """
    If f1, f2, ..., are functions which have no (required) arguments and
    return iterables, then
        lazy_product(f1, f2, ..., repeat=k)
    is equivalent to
        itertools.product(f1(), f2(), ..., repeat=k);
    but much faster in certain cases.
    For example, let f have the following definition:
        def f(n):
            def func():
                return xrange(n)
            return func
    Then, this code:
        p = itertools.product(*[f(N)() for _ in xrange(M)], repeat=K)
        first_element = next(p)
    takes O(NMK) time and memory to execute, whereas
        p = lazy_product(*[f(N) for _ in xrange(M)], repeat=K)
        first_element = next(p)
    is equivalent, and takes just O(MK) time and memory.
    (Of course, iterating over either result is exactly N^(MK) steps, and each
    step takes O(1) time; the only difference between itertools.product and
    lazy_product is at the time of initialization of the iterable p
    (including the call to next(p) to get the first element, as shown above).
    itertools.product's O(N) speed/memory overhead results from its saving the
    full result of xrange(N) as a list (or similar data structure) in memory.
    This is necessary as itertools.product takes iterables as input, and it is
    not generally possible to "reset" an iterator, so all of its values
    instead need to be stored. So, the input to lazy_product is an iterable
    of *functions* returning iterables, rather than the iterables themselves,
    allowing for repeated iteration over each iterable (by calling iter_func
    again when we reach the end of the iterable that iter_func created on
    the previous call).
    Inputs:
      - iter_funcs: functions with no (required) arguments that create and
        return an iterable. Each function is assumed to be be deterministic --
        i.e., return an identical iterable on each call.  (Otherwise, the
        behavior of lazy_product is undefined.)
      - kwargs: a dict which is either empty or contains only the key `repeat`,
        with an integer value.  In Python 3, the function header could (much
        more cleanly) be written as:
            def lazy_product(*iter_funcs, repeat=1):
        and the first two lines of ugly parsing code could be dropped.
    Returns:
        an iterator over the Cartesian product of the iterables returned
        by the elements of iter_funcs -- equivalent to:
            return itertools.product(*(f() for f in iter_funcs), **kwargs)
    """
    repeat = kwargs.pop('repeat', 1)
    if kwargs: raise ValueError('unknown kwargs: %s' % kwargs.keys())
    iters = [iter(f()) for _ in range(repeat) for f in iter_funcs]
    values = [next(i) for i in iters]
    while True:
        yield tuple(values)
        for index in reversed(range(len(iters))):
            try:
                values[index] = next(iters[index])
                break
            except StopIteration:
                iters[index] = iter(iter_funcs[index % len(iter_funcs)]())
                values[index] = next(iters[index])
        else: return


def delta_buchdahl_omega_bar(omega):
    '''
    Calculate the array $\\Delta \\bar{\\Omega}$ for running the de Albuquerque 
    glass selection process.
    '''
    n_min_1 = len(omega) - 1
    delta_omega_bar = np.zeros((n_min_1, n_min_1))
    for i_col in range(n_min_1):
        delta_omega_bar_col =  -np.diff(omega**(i_col+1.0))
        delta_omega_bar[:, i_col] = delta_omega_bar_col
    return delta_omega_bar

# For running de Albuquerque in parallel
import dask.distributed as dadi
# For iterating over glass combinations and group cartesian products
import itertools
# Logging for messages and such
import logging

class GlassCombo(object):
    """
    GlassCombo is a class for supporting the search for and analysis of glass combinations
    for e.g. the achromatisation of optical systems.

    A glass combination comprises a number of lens element groups. Each group comprises
    a number (generally a small number, but at least 2) of distinct glasses selected
    from a glass library, which may be different per group. The glass combinations
    that will be tested is the cartesion product of the glass combinations in each group.

    The class is capable of parallel execution. The number of combinations from each
    group is split over the number of processors. Suppose there are 4 processors.
    
    For correct load-balancing, the indices of the combinations for the first
    group are split into 4 subsets. Processor 1 will then generate one quarter
    of the combinations from group 1 and take the cartesion product with all the
    combinations from all the other groups. Likewise for the other three
    processors.

    The task scheduling across processors is performed by dask.distributed.
    
    """
    def __init__(self, wv, i_wv_0, gls_lib_per_grp, k_gls_per_grp, weight_per_grp, efl,
                    buchdahl_alpha=np.nan, sum_abs_pow_limit=10.0, max_result_rows=1048576,
                    do_opto_therm=True, temp_lo=20.0, temp_hi=21.0, pressure_env=101330.0,
                    max_delta_f=0.1,
                    show_progress=False, parallel=True, upstream=None):

        '''
        Build a case for computing best combinations of glasses for a first order lens layout,
        comprising a number of lens groups where the glasses for each group are taken
        from a glass library (ZemaxGlassLibrary) and where each group comprises a 
        fixed number of lens elements (each made of a distinct glass type, taken from
        the given library). Note that if two group glass libraries have glasses in common,
        then those common glasses can occur in both groups.

        Each lens group is idealised as a single thin (and flat) paraxial lens.

        A weight per group can be provided, generally with the first group having a weight
        of one, and for axial color, the weight is usually the marginal ray height at
        the group divided by the marginal ray height at the first group. These are the
        absolute paraxial marginal ray heights.

        For lateral color, the weight is related rather to the chief ray height at
        the paraxial thin lens group.

        Each group must comprise two or more distinct glasses, but the total number of
        glasses in the system is the sum of k_gls_per_grp, although there can
        be duplications between groups. It is recommended that glass catalog names be
        abbreviated using ZemaxGlass.abbreviate_cat_names() in order to reduce the memory
        requirements for large result tables.

        The glass combination search can be conducted with a fixed set up upstream glass 
        combinations. These will typically be the output of a previous glass
        search, and the downstream search is an effort to find good downstream
        glass combinations to match the promising upstream combinations. This process
        works by searching for downstream conbinations that have chromatic (and opto-thermal)
        residuals that cancel the residuals of the upstream combination. The upstream
        combinations have fixed power and weighting factor per glass in the combination.

        Parameters
        ----------
        wv : np array of float
            The wavelengths at which the system is to be corrected for chromatic aberration.
            Preferrably provided in nm, but will be considered as given in micron if
            less than 100. The number of wavelengths. At least three wavelengths must
            be provided.
        i_wv_0 : int
            The index of the reference wavelength in the wv array. That is, the reference
            wavelength must be one of the wavelengths provided, typically near the middle
            of the range.
        gls_lib_per_grp : list of ZemaxGlassLibrary
            The glass libraries to be used for each of the lens groups. The number of
            glasses in the library at least be equal to the number of distinct glasses
            to be assigned to the group (k_gls_per_grp). Abbreviate the  catalog names
            using ZmeaxGlass.abreviate_cat_names() to reduce result table memory.
        k_gls_per_grp : list of int
            The number of distinct glasses to be chosen at each of the lens groups.
            All list elements of k_gls_per_grp must be 2 or greater.
            Same list length as gls_lib_per_grp.
        weight_per_grp : numpy array of float
            The weights to assign to the lens groups. The weights must have physical
            meaning for the process to work. Since the process usually targets the
            axial color aberration, the ratio of marginal ray height at the lens group
            to the marginal ray height at the first group, which means that 
            weight_per_grp[0] = 1 very often.
        efl : float
            The focal length in mm of the lens under consideration in mm. This is used to
            scale the chromatic (and opto-thermal) defocus metrics. The reciprocal
            of the EFL expressed in metres is the total optical power of the 
            complete system under consideration (including upstream glass).
        buchdahl_alpha : float
            If the Buchdahl alpha parameter is known for the glass collection,
            this it can be provided here. Defaults to nan. Can be computed using
            the method buchdahl_find_alpha() or set using buchdahl_set_alpha()
            later.
        sum_abs_pow_limit : float
            Glass combinations with relative power distributions having a sum of
            absolute powers exceeding this limit will be discarded immediately
            after testing and will therefore not appear in the results dataframe.
            This is the first line of defence against producing very large result
            dataframes. Default is 10.0
        max_result_rows: int
            The maximum number of resulting evaluated glass combinations to return.
            Iteration over glass combinations will stop once this number of
            result rows have been inserted into the results dataframe. The 
            results with sum_abs_pow_lim exceeding the threshold will not be
            included in the count. Default is 2^20 = 1 048 576. This should
            not be regarded as exact, just there to prevent interminable runs.
        do_opto_therm : boolean
            If set True, the opto-thermal coefficient for the system will be
            computed, assuming the expansion coefficient of the barrel 
            material is zero. If the delta_temp input is also given, the
            opto-thermal defocus will be calculated for the given
            total temperature swing. Default is True.
        temp_lo : float
            The minimum environmental operating temperature to consider for
            opto-thermal calculations. In deg C. Default 20.0,
        temp_hi : float
            The maximum environmental operating temperature to consider for
            opto-thermal calculations. In deg C. Default 21.0.
        pressure_env : float
            The environmental pressure in Pa. Default 1 atm = 101330.0 Pa.
        show_progress : boolean
            If set True, in an IPython notebook environment, a simple
            text progress bar will be displayed.
        parallel : boolean
            If set True, the code will attempt to run the search in parallel
            on all available processors. The load balancing may not be
            ideal, but substantial speedup should occur. Default True.
            Note that this feature uses the dask.distributed Python module.
        upstream : GlassCombo object
            Upstream glass combinations to be included in the search for
            optimal downstream glass combinations. The best glass combination
            results are used as good candidate upstream glass combinations
            The GlassCombo passed in as upstream, must have selected best results
            stored in the attribute best_gls_combos_df.
            Default is None, in which case there will be no upstream glass
            combos targeted for residual chromatic or thermo-chromatic
            correction.

        '''

        # Perform a few simple checks
        if len(wv) < 3:
            raise ValueError('Input wv must be a numeric vector of at least 3 floating point wavelength values.')
        if len(gls_lib_per_grp) != len(k_gls_per_grp):
            raise ValueError('Input gls_lib_per_grp must be a list of the same length as the k_gls_per_grp input list.')
        if len(k_gls_per_grp) != len(weight_per_grp):
            raise ValueError('Input weight_per_grp must be a list of the same length as the k_gls_per_grp input list.')
        self.num_grp = len(gls_lib_per_grp)  # Number of lens groups in the problem
        # Calculate the total number of combinations to be tested
        # First fetch the number of glasses in each library
        num_gls_per_grp = [0] * len(gls_lib_per_grp)
        comb_per_grp = [0] * len(gls_lib_per_grp)
        for i_grp, glass_lib in enumerate(gls_lib_per_grp):
            num_gls_per_grp[i_grp] = glass_lib.get_num_glasses()
            comb_per_grp[i_grp] = combinations(num_gls_per_grp[i_grp], k_gls_per_grp[i_grp])
        # Deal with upstream glass combos
        if upstream is not None:
            self.process_upstream_combos(upstream)
            self.total_combinations = np.array(comb_per_grp, dtype=np.int64).prod() * self.n_upstream_combos
        else:
            # This is the cartesian product of the combinations for each group, could be very large
            self.total_combinations = np.array(comb_per_grp, dtype=np.int64).prod()
        # Gentle reminder to the user about the number of combinations they could be up against
        self.filename = datetime.now().strftime('%Y%m%d-%H%M%S')
        runtime = int(self.total_combinations//5000)
        print(f'Take note that the number of potential glass combinations in this instance is {self.total_combinations}.')
        print(f'Depending on available hardware and parallel execution, exhaustive processing could take a long time.')
        print(f'Order of magnitude run time is {str(timedelta(seconds=runtime))} divided by the number of cores. ')
        print(f'Processing time can be reduced by reducing the number of glasses available for each group.')
        print(f'One approach is to first drastically narrow down the glass selections for each group in turn.')
        print(f'Intermediate results (if any) will be stored in files, name starting {self.filename} with suitable extension.')
        # Record all other information in the instance
        #
        self.wv = wv
        self.i_wv_0 = i_wv_0
        self.num_wv = wv.size
        self.wv_0 = wv[i_wv_0]
        self.num_gls_per_grp = num_gls_per_grp        
        self.comb_per_grp = comb_per_grp  # Total number of potential combinations for each group
        self.k_gls_per_grp = k_gls_per_grp
        self.k_gls = sum(k_gls_per_grp)  # Total number of glasses per combination, all groups
        if self.k_gls < 2:
            raise ValueError('Total number of glasses to choose must be at least 2.')
        self.gls_lib_per_grp = gls_lib_per_grp
        self.weight_per_grp = weight_per_grp
        self.max_delta_f = max_delta_f
        self.max_result_rows = max_result_rows
        self.max_result_rows_per_worker = np.nan  # Not yet known, will be computed when client is set up
        self.sum_abs_pow_limit = sum_abs_pow_limit  # Maximum sum of absolute powers per combination (all groups)
        self.do_opto_therm = do_opto_therm
        self.gamma_per_grp = []  # Opto-thermal coefficients per group
        self.get_all_cat_gls()  # Obtains cat_all, gls_all, cat_per_grp and gls_per_grp
        # Thermal stuff
        self.temp_hi = temp_hi
        self.temp_lo = temp_lo
        self.pressure_env = pressure_env
        self.delta_temp = temp_hi - temp_lo        
        if do_opto_therm:  # Calculate opto-thermal coefficients for all materials
            self.get_all_opto_therm_coeff()
        self.show_progress = show_progress
        self.efl = efl
        self.parallel = parallel
        self.buchdahl_alpha = buchdahl_alpha 
        # Calculate the buchdahl omega coordinates, will be nan if buchdahl_alpha is nan
        self.omega = buchdahl_omega(self.wv, self.wv_0, self.buchdahl_alpha)
        # The following are constant (per GlassCombo problem) de Albuquerque vectors/matrices
        self.delta_omega_bar = delta_buchdahl_omega_bar(self.omega)
        self.big_s_bar = self.calc_big_s_bar()  # Row vector of weights, adapted from de Albuquerque
        self.big_w_bar = self.calc_big_w_bar()  # This is currently the same as big_s_bar
        # $\hat{e}$ is a column vector with number of elements equal to the number of wavelengths,
        # having 1.0 at the top and all zeros below
        self.e_hat = np.vstack((np.array([1.0]), np.zeros((self.num_wv - 1, 1))))
        self.buchdahl_alphas = np.empty(len(self.gls_all)) * np.nan  # not yet known
        # Any dask.distributed setup here
        self.worker_iterators = None  # These produce the glass combination indices
        # Run statistics, populated after de Albuquerque run completes
        self.last_run_total_results = None
        self.last_run_percent_max_results = None
        # The following attribute is used to store best glass combo results for
        # this instance. The attribute is populated using 
        self.best_gls_combos_df = None
  
    def process_upstream_combos(self, upstream):
        '''
        Process the upstream GlassCombo instance, mainly to extract the relevant upstream
        residuals.

        Parameters
        ----------
        upstream : instance of GlassCombo 
            This GlassCombo upstream input represents the best results from a previous
            glass combination search. It must have the best_gls_combo_df attribute
        '''
        # Run through groups and set up the column names 
        for i_grp in range(upstream.num_grp):
            weight = upstream.weight_per_grp[i_grp]
            j_gls = 0
            for k_gls in range(upstream.k_gls_per_grp[i_grp]):
                j_gls += 1
                cat_col = f'c{j_gls}'
                gls_col = f'g{j_gls}'
                pow_col = f'p{j_gls}'
                wgt_col = f'w{j_gls}'
                print(cat_col, gls_col, pow_col)
                # Iterate over the rows of the dataframe
                for _, combo_row in upstream.best_gls_combos_df.iterrows():
                    print(combo_row[cat_col], combo_row[gls_col], combo_row[pow_col])
        self.n_upstream_combos = len(upstream.best_gls_combos_df)

    def save_best_combos(self, best_gls_combos_df):
        '''
        Saves the best glass combinations for later use, most importantly as good upstream 
        glass combinations for a new GlassCombo object that will continue with the downstream
        glass combination search. This should be a filtered and relatively short dataframe
        of glass combinations returned for the search executed on THIS instance of the
        GlassCombo class.

        Parameters
        ----------
        best_gls_combos_df : pandas dataframe of glass combinations
            This is a dataframe of the kind returned by run_de_albuquerque() method.
            It must include columns cn, gn, pn, where n is an integer starting at 1.
            These columns are respectively the catalog, glass and absolute power
            of the glass
        '''
        # Save the best candidates and calculate weights and residuals
        # Extract mandatory columns, adding weight column
        self.best_gls_combos_df = pd.DataFrame()
        n_best_gls_combos = len(best_gls_combos_df)
        j_gls = 0
        for i_grp in range(self.num_grp):
            weight = self.weight_per_grp[i_grp]
            for _ in range(self.k_gls_per_grp[i_grp]):
                j_gls += 1
                cat_col = f'c{j_gls}'
                gls_col = f'g{j_gls}'
                pow_col = f'p{j_gls}'
                wgt_col = f'w{j_gls}'
                self.best_gls_combos_df[cat_col] = best_gls_combos_df[cat_col]
                self.best_gls_combos_df[gls_col] = best_gls_combos_df[gls_col]
                self.best_gls_combos_df[pow_col] = best_gls_combos_df[pow_col]
                self.best_gls_combos_df[wgt_col] = [weight] * n_best_gls_combos
        self.n_best_gls_combos = n_best_gls_combos 
 

    def calc_big_s_bar(self):
        '''
        Calculate the $\\bar{S}$ vector for the de Albuquerque method. This is a row
        vector of length $k$, the total number of glasses `k_gls`. The values correspond
        to the weights `weight_per_grp` provided in the call to the constructor.
        '''
        big_s_bar = []
        for i_grp in range(self.num_grp):
            grp_weight = [self.weight_per_grp[i_grp]] * self.k_gls_per_grp[i_grp]
            big_s_bar.extend(grp_weight)
        return np.array(big_s_bar)


    def calc_big_w_bar(self):
        '''
        Calculate the $\\bar{W}$ vector for the de Albuquerque method. This is a row
        vector of length $k$, the total number of glasses `k_gls`. The values correspond
        to the weights `weight_per_grp` provided in the call to the constructor.
        The $\\bar{W}$ vector takes over from the standard $\\bar{S}$ vector.
        '''
        big_w_bar = []
        for i_grp in range(self.num_grp):
            grp_weight = [self.weight_per_grp[i_grp]] * self.k_gls_per_grp[i_grp]
            big_w_bar.extend(grp_weight)
        return np.array(big_w_bar)

   
    def get_all_cat_gls(self):
        '''
        Get pair-wise (same length) lists of all the catalogs and glasses in
        the combo.
        Also provides the catalogs and glasses per group in the attributes
        cat_per_grp and gls_per_grp.
        '''
        cat = []
        gls = []
        self.cat_per_grp = []
        self.gls_per_grp = []     
        for gls_lib in self.gls_lib_per_grp:
            cats, glss = gls_lib.get_all_cat_gls()
            self.cat_per_grp.append(cats)
            self.gls_per_grp.append(glss)
            cat.extend(cats)
            gls.extend(glss) 
        self.cat_all = cat
        self.gls_all = gls
        return cat, gls

    def get_all_opto_therm_coeff(self):
        '''
        Compute all opto-thermal coefficients and retrieve as a vector.
        Need to be careful when this is called. The glass libraries are
        dicts and the order in which the catalogs and glasses are 
        processed could be different. This process uses the order
        in which they were last returned to self.cat_per_grp and
        self.gls_per_grp.
        '''
        gamma_per_grp = [np.array([])]*self.num_grp
        for i_grp in range(self.num_grp):
            self.gls_lib_per_grp[i_grp].add_opto_thermal_coeff(self.temp_lo, self.temp_hi, self.wv_0, self.pressure_env)
            for cat, gls in zip(self.cat_per_grp[i_grp], self.gls_per_grp[i_grp]):
                gamma_per_grp[i_grp] = np.append(gamma_per_grp[i_grp], 
                                    self.gls_lib_per_grp[i_grp].library[cat][gls]['opto_therm_coeff'])
        self.gamma_per_grp = gamma_per_grp

    def buchdahl_find_alpha(self, show_progress=False):
        '''
        Determine the mean best-fit Buchdahl alpha parameter for ALL the glasses
        in the libraries associated with all groups.

        This method will also calculate or recalculate the Buchdahl omega
        coordinates.

        Parameters
        ----------
        show_progress : boolean
            If set True, will display a simple text progress bar - useful for 
            notebook environment. However, the progress bar is reset for
            each glass library in the instance. At least you get to see that
            progress is being made.
        
        Returns
        -------
        None, the mean best-fit alpha buchdahl parameter is stored in the
        class attribute buchdahl_alpha. The best fit alphas
        '''
        buchdahl_alphas = np.array([])
        for gls_lib in self.gls_lib_per_grp:
            _, _, buch_fits = gls_lib.buchdahl_find_alpha(self.wv, self.wv_0, show_progress=show_progress)
            buchdahl_alphas = np.hstack((buchdahl_alphas, buch_fits[:, 0]))
        self.buchdahl_alphas = buchdahl_alphas
        self.buchdahl_alpha = buchdahl_alphas.mean()
        self.get_all_cat_gls()  # update lists in case they have changed
        # And calculate or recalculate the Buchdahl omega coordinates
        self.omega = buchdahl_omega(self.wv, self.wv_0, self.buchdahl_alpha)
        self.delta_omega_bar = delta_buchdahl_omega_bar(self.omega)
    
    def buchdahl_set_alpha(self, buchdahl_alpha):
        '''
        Manually set the Buchdahl alpha parameter for all the glasses in the combo.
        It is better to find the optimum value using buchdahl_find_alpha().
        However, that is quite time-consuming, so if the value is already known,
        it can be set using this method.

        The Buchdahl omega coordinates will also be calculated ot recalculated.
        '''
        self.buchdahl_alpha = buchdahl_alpha
        # Also compute the Buchdahl omega coordinates for the problem
        self.omega = buchdahl_omega(self.wv, self.wv_0, self.buchdahl_alpha)
        self.delta_omega_bar = delta_buchdahl_omega_bar(self.omega)        

    def buchdahl_fit_eta(self, show_progress=False):
        '''
        Find the Buchdahl eta coefficients for all the glasses in the combo.
        These are the coefficients of the Buchdahl dispersive power function,
        not the dispersion function. The results are computed using the
        ZemaxGlassLibrary.buchdahl_fit_eta() method.
        '''
        self.cat_per_grp = []
        self.gls_per_grp = []
        self.eta_per_grp = []
        self.n_ref_per_grp = []
        if np.isnan(self.buchdahl_alpha):
            raise ValueError('The Buchdahl alpha parameter must first be set for this combo using buchdahl_find_alpha() or buchdahl_set_alpha().')
        for gls_lib in self.gls_lib_per_grp:
            cat, gls, eta, n_ref = gls_lib.buchdahl_fit_eta(self.wv, self.i_wv_0, alpha=self.buchdahl_alpha,
                                                                show_progress=show_progress)
            self.cat_per_grp.append(cat)
            self.gls_per_grp.append(gls)
            self.eta_per_grp.append(eta)
            self.n_ref_per_grp.append(n_ref)
        self.eta_per_grp = np.array(self.eta_per_grp)
        self.n_ref_per_grp = np.array(self.n_ref_per_grp)
        if self.do_opto_therm:  # Refresh the opto-thermal coefficents
            self.get_all_opto_therm_coeff()

    # Utility methods for working with dask.distributed, imported as dadi

    def dask_client_setup(self, dask_client=None, dask_scheduler=None, num_workers=None):
        '''
        Adds a dask.distributed compute client to the self instance.
        Also determines how to partition the problem across the number
        of available workers.

        Parameters
        ----------
        dask_scheduler : str
            IPnumber:port of the dask.distributed scheduler.
            Default is None, which means the scheduler will
            be started on the local machine. The number of 
            workers is set in self according to the number of
            workers available reported by the scheduler.
        num_workers : int
            Overrides number of workers reported by the dask scheduler.
            This is the number of worker processors that will be started.

        Returns
        -------
        dask_client : dask.distributed.Client
            The dask client to use for execution of combination search
            tasks on this GlassCombo.
        '''
        if dask_client is None:
            if dask_scheduler is None:
                dask_client = dadi.Client()
            else:
                dask_client = dadi.Client(dask_scheduler)
        if num_workers is not None:
            self.dask_num_workers = num_workers
        else:
            self.dask_num_workers = len(dask_client.scheduler_info()['workers'])
        # Get the index of lens group with the maximum number of possible glass combinations
        self.dask_get_comb_split_among_workers()  # 
        self.max_result_rows_per_worker = self.max_result_rows // self.dask_num_workers
        # Produce the iterators
        # self.dask_produce_iterators(dask_client)        
        return dask_client

    def dask_get_comb_split_among_workers(self):
        '''
        Determine the glass combination index splits on the lens group with
        the maximum number of combinations. These are the indices that are
        passed to the nth_combination function to generate the sequence of
        glass combination indices.
        '''
        # Get the index of lens group with the maximum number of possible glass combinations
        self.i_grp_max_combo = self.comb_per_grp.index(max(self.comb_per_grp))
        self.grp_max_combo = self.comb_per_grp[self.i_grp_max_combo]
        # Get the distribution amongst the available number of workers
        splits, ranges = split_combos_m_ways(self.num_gls_per_grp[self.i_grp_max_combo], 
                                    self.k_gls_per_grp[self.i_grp_max_combo],
                                    self.dask_num_workers)
        self.comb_ind_worker_splits = splits
        self.comb_ind_worker_ranges = ranges

    def dask_client_restart(self, dask_client):
        '''
        Restarts the dask.distributed client given.
        ''' 
        # Recreate the iterators
        self.dask_produce_iterators(dask_client)    
        dask_client.restart()
    
    @staticmethod
    def dask_client_info(dask_client):
        '''
        Returns a dict with a wealth of information on the dask.distributed client
        and associated workers.
        '''
        return dask_client.scheduler_info()
    
    @staticmethod
    def dask_get_dashboard_url(dask_client):
        '''
        Gets the URL of the bokeh dashboard for the dask.distributed client (if present).
        '''
        info = dask_client.scheduler_info()
        address = info['address']
        services = info['services']
        if 'dashboard' in services.keys():
            dashport = services['dashboard']  # port of the dashboard bokeh service
        else:
            return None
        # Formulate the URL of the dashboard
        re_url = r"^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)"
        match = re.match(re_url, address)
        ip_num = match.groups()[1]
        dashboard_url = 'http://' + ip_num + ':' + str(dashport)
        return dashboard_url
    
    @staticmethod
    def dask_open_dashboard(dask_client, new=2):
        '''
        Open a browser on the dask.distributed dashboard.
        
        Parameters
        ----------
        new : int
            Default is 2, which means open a new tab if the default browser
            is already running.
        '''
        dashboard_url = GlassCombo.dask_get_dashboard_url(dask_client)
        if dashboard_url is not None:
            import webbrowser
            webbrowser.open(dashboard_url, new=new)

    def build_eta_bar(self, combo):
        '''
        Put together the Buchdahl eta coefficients for a specific glass combination.
        Also retrieve the glass and catalog and names for the combination.

        Parameters
        ----------
        combo : list of tuples
            List of 2-tuples providing the indices of the glasses for the combination
            being assessed. The number of elements in each tuple must correspond to
            the number of glasses in each group and the number of tuples must
            correspond to the number of groups.
        
        Returns
        -------
        combo_cat : list of str
            Glass catalogs for the glasses in the combination
        combo_gls : list of str
            Glass names for the glasses in the combination 
        eta_bar : ndarray of float
            Buchdahl dispersive power eta coefficients for all the glasses in the
            combination, one column per glass, and number of rows `wv_num - 1`.
        '''
        eta_bar = self.eta_per_grp[0][list(combo[0]), :]  # There must be at least one combo
        combo_cat = [self.cat_per_grp[0][i_cat] for i_cat in combo[0]]
        combo_gls = [self.gls_per_grp[0][i_cat] for i_cat in combo[0]]
        for i_grp in range(1, self.num_grp):
            eta_bar = np.vstack((eta_bar, self.eta_per_grp[i_grp][list(combo[i_grp]), :]))
            combo_cat.extend([self.cat_per_grp[i_grp][i_cat] for i_cat in combo[i_grp]])
            combo_gls.extend([self.gls_per_grp[i_grp][i_cat] for i_cat in combo[i_grp]])
        return combo_cat, combo_gls, eta_bar.T
    
    def build_big_h_bar(self, combo):
        '''
        Put together the weighted Buchdahl eta coefficients for a specific glass combination.

        Also retrieve the glass and catalog and names for the combination.

        Parameters
        ----------
        combo : list of tuples
            List of 2-tuples providing the indices of the glasses for the combination
            being assessed. The number of elements in each tuple must correspond to
            the number of glasses in each group and the number of tuples must
            correspond to the number of groups.
        
        Returns
        -------
        combo_cat : list of str
            Glass catalogs for the glasses in the combination
        combo_gls : list of str
            Glass names for the glasses in the combination 
        big_h_bar : ndarray of float
            Weighted Buchdahl dispersive power eta coefficients for all the glasses in the
            combination, one column per glass, and number of rows `wv_num - 1`.
            Glass columns are weighted by the group weights in the instance.
        '''
        big_h_bar = self.weight_per_grp[0] * self.eta_per_grp[0][list(combo[0]), :]  # There must be at least one combo
        combo_cat = [self.cat_per_grp[0][i_cat] for i_cat in combo[0]]
        combo_gls = [self.gls_per_grp[0][i_cat] for i_cat in combo[0]]
        for i_grp in range(1, self.num_grp):
            big_h_bar = self.weight_per_grp[i_grp] * np.vstack((big_h_bar, self.eta_per_grp[i_grp][list(combo[i_grp]), :]))
            combo_cat.extend([self.cat_per_grp[i_grp][i_cat] for i_cat in combo[i_grp]])
            combo_gls.extend([self.gls_per_grp[i_grp][i_cat] for i_cat in combo[i_grp]])
        return combo_cat, combo_gls, big_h_bar.T         

    def build_gamma(self, combo):
        '''
        Build a row vector of opto-thermal coefficients $\\gamma$ for the glasses in
        the combination.
        '''
        gamma = []
        for i_grp in range(self.num_grp):
            for i_gls in combo[i_grp]:
                gamma.append(self.gamma_per_grp[i_grp][i_gls]) 
        return np.array(gamma)

    def build_big_gamma(self, combo):
        '''
        Build a row vector of weighted opto-thermal coefficients $\\Gamma$ for the glasses in
        the combination.
        '''
        big_gamma = []
        for i_grp in range(self.num_grp):
            for i_gls in combo[i_grp]:
                big_gamma.append(self.weight_per_grp[i_grp] * self.gamma_per_grp[i_grp][i_gls]) 
        return np.array(big_gamma)

    def dask_run_worker_with_upstream(self, i_worker):
        '''
        Run the de Albuquerque method on a range of group/glass combinations produced by
        an iterator. This is where the heavy computation is performed.
        This version of the worker will include upstream glass combination
        residual targeting.

        Returns
        -------
        A pandas or dask dataframe of results.
        '''        
        # Allocate output space for the maximum number of results per worker for this run
        # It should be (or become) possible to call this again for another set of results,
        # provided that the worker_iterators are not reset.

        combo_index_range = self.comb_ind_worker_ranges[i_worker]  # 2-tuple of start end combo indices
        worker_iterator_max_combo = gen_combination(combo_index_range[0], combo_index_range[1], 
                                            range(self.num_gls_per_grp[self.i_grp_max_combo]),
                                            self.k_gls_per_grp[self.i_grp_max_combo])
        max_combo_per_worker = self.comb_ind_worker_splits[i_worker]  # gets multiplied by unsplit group combos
        # For each worker build a cartesian product iterator across lens groups
        grp_iterators = [None]*self.num_grp

        for i_grp in range(self.num_grp):
            # print(f'Producing Iterator for Group # {i_grp}')                
            if i_grp == self.i_grp_max_combo:
                grp_iterators[i_grp] = worker_iterator_max_combo
            else:
                # This is an ordinary (unsplit) iterator over combinations for the lens group
                max_combo_per_worker *= self.comb_per_grp[i_grp]
                grp_iterators[i_grp] = itertools.combinations(range(self.num_gls_per_grp[i_grp]),
                                                self.k_gls_per_grp[i_grp])
        worker_iterator = itertools.product(*grp_iterators)  # Cartesian product of iterators
        # Calculate the target optical power for this run, this will be the first element in vector e_bar
        big_phi_0 = 1000.0 / self.efl  # dioptre, because efl assumed to be in mm
        e_bar = self.e_hat * big_phi_0  # target vector
        # Calculate size of arrays to initialise for storage, limited to max_combo_per_worker
        max_i_combo = min(self.max_result_rows_per_worker, max_combo_per_worker)
        abso_pow = np.zeros((max_i_combo, self.k_gls))  # Absolute powers assigned to each glass for best correction
        sum_abs_norm_pow = np.zeros(max_i_combo)  # Summation of the absolute normalised glass powers for a combo
        norm_chroma_pow_delta = np.zeros(max_i_combo)  # Modulus of the normalised chromatic power shift F_2 = |CCP|
        therm_power_rate = np.zeros(max_i_combo)  # Rate of change of total optical power with temperature
        delta_temp_F = np.zeros(max_i_combo)  # Total absolute focal shift over temperature range delta_temp
        delta_color_F = np.zeros(max_i_combo)  # Absolute focal shift over wavelength |CCP| * F
        delta_F = np.zeros(max_i_combo)  # RSS delta_temp_F and delta_color_F        
        worker_gls = []  # Accumulate the glasses in the main loop(s)
        worker_cat = []  # Accumulate corresponding glass catalogues in the main loop(s)
        # combos = []  # Will also grow a list of the combos processed, waste of time and memory?
        # Loop over the iterator until the maximum number of combinations results is obtained
        # Or the interator is exhausted
        i_combo = 0  # This counts the number of glass combinations processed by this worker in this run
        combos_all_done = True  # Will be set False later if not all done
        for combo in worker_iterator:
            # combos.append(combo)
            # Build the big_h_bar matrix, one column per glass, weighted, also obtain catalogs and glasses
            combo_cat, combo_gls, big_h_bar = self.build_big_h_bar(combo)
            # Build row vector of weighted opto-thermal coefficients for glasses in combo
            big_gamma_gls = self.build_big_gamma(combo)  # Length is total glasses per combo
            # Calculate the big_g_bar matrix
            big_g_bar = np.vstack((self.big_w_bar, np.matmul(self.delta_omega_bar, big_h_bar)))
            # Calculate the determinat of big_g_bar * transpose(big_g_bar)
            # The glass combination is sometimes degenerate because one or more
            # glasses in the combo are identical or nearly so
            determinant = np.linalg.det(np.matmul(big_g_bar.T, big_g_bar))
            # Ditch this combination if the determinant is zero - bad combo
            if determinant == 0.0:
                continue         
            # Calculate the mashup of the big_g_bar matrix, much happens in this line
            big_g_bar_mash = np.matmul(np.linalg.inv(np.matmul(big_g_bar.T, big_g_bar)), big_g_bar.T)
            # Now run through all upstream combos with this downstream combo
            for upstream_combo in self.upstream_combos:
                # Set up the e_bar target vector

                # Calculate the big_phi_bar matrix (optimal dioptre powers per glass)
                big_phi_bar = np.matmul(big_g_bar_mash, e_bar)
                # Calculate the chromatic change of absolute (dioptre) power per wavelength 
                # These are the chromatic residuals, absolute power version of vector called CCP by de Albuquerque
                chroma_power_delta = np.matmul(self.delta_omega_bar, np.matmul(big_h_bar, big_phi_bar))
                # Calculate the chromatic focal shift by multiplying by the effective focal length (mm)
                delta_color_F[i_combo] = np.linalg.norm(-self.efl**2.0 * chroma_power_delta/1.0e3)  # Effectively |CCP| * F
                # Save the data, chromatic focal shift, power distributions
                abso_pow[i_combo, :] = big_phi_bar.T  # Record the absolute (dioptre) power for the 3-glass combo            
                # If the sum of absolute normalised powers exceeds threshold
                # abort further processing and don't record the result
                sum_abs_norm_pow[i_combo] = np.abs(abso_pow[i_combo, :]/big_phi_0).sum()
                if sum_abs_norm_pow[i_combo] > self.sum_abs_pow_limit:
                    continue
                # Focus shift over whole temperature range in mm (if EFL is in mm)
                delta_temp_F[i_combo] = - self.efl**2.0 * self.delta_temp * (big_phi_bar.T * big_gamma_gls).sum() / 1.0e3
                # RSS chromatic and opto-thermal focus shifts
                delta_F[i_combo] = np.sqrt(delta_temp_F[i_combo]**2.0 + delta_color_F[i_combo]**2.0)
                norm_chroma_pow_delta[i_combo] = np.linalg.norm(chroma_power_delta)  # |CCP|                       
                # Record cats and glass for the combo if past this point          
                worker_cat.append(combo_cat)
                worker_gls.append(combo_gls)               
                # If the maximum number of rows has been reached, stop and return results
                i_combo += 1
                if i_combo == max_i_combo:
                    combos_all_done = False  # Never got through all combinations, so coverage is incomplete
                    break
        # Build dataframe of all relevant results
        # Build dataframe column names for glasses and catalogs
        gls_df_cols = [f'g{i_gls+1}' for i_gls in range(self.k_gls)]
        cat_df_cols = [f'c{i_gls+1}' for i_gls in range(self.k_gls)]
        cat_df = pd.DataFrame(worker_cat, dtype='category')
        cat_df.columns = cat_df_cols        
        gls_df = pd.DataFrame(worker_gls, dtype='category')
        gls_df.columns = gls_df_cols
        cat_gls_df = pd.concat([cat_df, gls_df], axis=1)
        abs_pow_cols = [f'p{i_gls+1}' for i_gls in range(self.k_gls)]
        abs_pow_df = pd.DataFrame(abso_pow[0:i_combo, :])
        abs_pow_df.columns = abs_pow_cols
        worker_dataframe = pd.concat([cat_gls_df, abs_pow_df], axis=1)
        # At this point reorder the columns so that the power values are interleaved
        # with the catalog/glass columns
        cat_gls_pow_cols = []
        for i_gls in range(self.k_gls):
            cat_gls_pow_cols.extend([f'c{i_gls+1}', f'g{i_gls+1}', f'p{i_gls+1}'])
        worker_dataframe = worker_dataframe[cat_gls_pow_cols]
        # Add the sum of normalised powers
        worker_dataframe['f_1'] = sum_abs_norm_pow[0:i_combo]
        # Add the norm of the |CCP| * efl
        worker_dataframe['f_2'] = delta_color_F[0:i_combo]
        # Skip de Albuquerque F3, which is sum of some aberration residuals in aplanatic arrangement
        # Add the absolute thermal change in focus over the full delta_temp
        worker_dataframe['f_4'] = delta_temp_F[0:i_combo]
        # Any dataframe metadat/attributes
        worker_dataframe.attrs['combos_all_done'] = combos_all_done
        return worker_dataframe

    def dask_run_worker(self, i_worker):
        '''
        Run the de Albuquerque method on a range of group/glass combinations produced by
        an iterator. This is where the heavy computation is performed. 

        Returns
        -------
        A pandas or dask dataframe of results.
        '''        
        # Allocate output space for the maximum number of results per worker for this run
        # It should be (or become) possible to call this again for another set of results,
        # provided that the worker_iterators are not reset.

        combo_index_range = self.comb_ind_worker_ranges[i_worker]  # 2-tuple of start end combo indices
        worker_iterator_max_combo = gen_combination(combo_index_range[0], combo_index_range[1], 
                                            range(self.num_gls_per_grp[self.i_grp_max_combo]),
                                            self.k_gls_per_grp[self.i_grp_max_combo])
        max_combo_per_worker = self.comb_ind_worker_splits[i_worker]  # gets multiplied by unsplit group combos
        # For each worker build a cartesian product iterator across lens groups
        grp_iterators = [None]*self.num_grp

        for i_grp in range(self.num_grp):
            # print(f'Producing Iterator for Group # {i_grp}')                
            if i_grp == self.i_grp_max_combo:
                grp_iterators[i_grp] = worker_iterator_max_combo
            else:
                # This is an ordinary (unsplit) iterator over combinations for the lens group
                max_combo_per_worker *= self.comb_per_grp[i_grp]
                grp_iterators[i_grp] = itertools.combinations(range(self.num_gls_per_grp[i_grp]),
                                                self.k_gls_per_grp[i_grp])
        worker_iterator = itertools.product(*grp_iterators)  # Cartesian product of iterators
        # Calculate the target optical power for this run, this will be the first element in vector e_bar
        big_phi_0 = 1000.0 / self.efl  # dioptre, because efl assumed to be in mm
        e_bar = self.e_hat * big_phi_0  # target vector
        # Calculate size of arrays to initialise for storage, limited to max_combo_per_worker
        max_i_combo = min(self.max_result_rows_per_worker, max_combo_per_worker)
        abso_pow = np.zeros((max_i_combo, self.k_gls))  # Absolute powers assigned to each glass for best correction
        sum_abs_norm_pow = np.zeros(max_i_combo)  # Summation of the absolute normalised glass powers for a combo
        norm_chroma_pow_delta = np.zeros(max_i_combo)  # Modulus of the normalised chromatic power shift F_2 = |CCP|
        therm_power_rate = np.zeros(max_i_combo)  # Rate of change of total optical power with temperature
        delta_temp_F = np.zeros(max_i_combo)  # Total absolute focal shift over temperature range delta_temp
        delta_color_F = np.zeros(max_i_combo)  # Absolute focal shift over wavelength |CCP| * F
        delta_F = np.zeros(max_i_combo)  # RSS delta_temp_F and delta_color_F        
        worker_gls = []  # Accumulate the glasses in the main loop(s)
        worker_cat = []  # Accumulate corresponding glass catalogues in the main loop(s)
        # combos = []  # Will also grow a list of the combos processed, waste of time and memory?
        # Loop over the iterator until the maximum number of combinations results is obtained
        # Or the interator is exhausted
        i_combo = 0  # This counts the number of glass combinations processed by this worker in this run
        combos_all_done = True  # Will be set False later if not all done
        for combo in worker_iterator:
            # combos.append(combo)
            # Build the big_h_bar matrix, one column per glass, weighted, also obtain catalogs and glasses
            combo_cat, combo_gls, big_h_bar = self.build_big_h_bar(combo)
            # Build row vector of weighted opto-thermal coefficients for glasses in combo
            big_gamma_gls = self.build_big_gamma(combo)  # Length is total glasses per combo
            # Calculate the big_g_bar matrix
            big_g_bar = np.vstack((self.big_w_bar, np.matmul(self.delta_omega_bar, big_h_bar)))
            # Calculate the determinat of big_g_bar * transpose(big_g_bar)
            # The glass combination is sometimes degenerate because one or more
            # glasses in the combo are identical or nearly so
            determinant = np.linalg.det(np.matmul(big_g_bar.T, big_g_bar))
            # Ditch this combination if the determinant is zero - bad combo
            if determinant == 0.0:
                continue         
            # Calculate the mashup of the big_g_bar matrix, much happens in this line
            big_g_bar_mash = np.matmul(np.linalg.inv(np.matmul(big_g_bar.T, big_g_bar)), big_g_bar.T)
            # Calculate the big_phi_bar matrix (optimal dioptre powers per glass)
            big_phi_bar = np.matmul(big_g_bar_mash, e_bar)
            # Calculate the chromatic change of absolute (dioptre) power per wavelength 
            # These are the chromatic residuals, absolute power version of vector called CCP by de Albuquerque
            chroma_power_delta = np.matmul(self.delta_omega_bar, np.matmul(big_h_bar, big_phi_bar))
            # Calculate the chromatic focal shift by multiplying by the square of the effective focal length (mm)
            delta_color_F[i_combo] = np.linalg.norm(-self.efl**2.0 * chroma_power_delta/1.0e3)  # Effectively |CCP| * F
            # Save the data, chromatic focal shift, power distributions
            abso_pow[i_combo, :] = big_phi_bar.T  # Record the absolute (dioptre) power for the combo            
            # If the sum of absolute normalised powers exceeds threshold
            # abort further processing and don't record the result
            sum_abs_norm_pow[i_combo] = np.abs(abso_pow[i_combo, :]/big_phi_0).sum()
            if sum_abs_norm_pow[i_combo] > self.sum_abs_pow_limit:
                continue
            # Focus shift over whole temperature range in mm (if EFL is in mm)
            delta_temp_F[i_combo] = - self.efl**2.0 * self.delta_temp * (big_phi_bar.T * big_gamma_gls).sum() / 1.0e3
            # RSS chromatic and opto-thermal focus shifts
            delta_F[i_combo] = np.sqrt(delta_temp_F[i_combo]**2.0 + delta_color_F[i_combo]**2.0)
            if delta_F[i_combo] > self.max_delta_f:
                continue
            norm_chroma_pow_delta[i_combo] = np.linalg.norm(chroma_power_delta)  # |CCP|                       
            # Record cats and glass for the combo if past this point          
            worker_cat.append(combo_cat)
            worker_gls.append(combo_gls)               
            # If the maximum number of rows has been reached, stop and return results
            i_combo += 1
            if i_combo == max_i_combo:
                combos_all_done = False  # Never got through all combinations, so coverage is incomplete
                break
        if i_combo == 0:
            return   # This worker found nothing within the ambit of the search
        # Build dataframe of all relevant results
        # Build dataframe column names for glasses and catalogs
        gls_df_cols = [f'g{i_gls+1}' for i_gls in range(self.k_gls)]
        cat_df_cols = [f'c{i_gls+1}' for i_gls in range(self.k_gls)]
        cat_df = pd.DataFrame(worker_cat, dtype='category')
        cat_df.columns = cat_df_cols        
        gls_df = pd.DataFrame(worker_gls, dtype='category')
        gls_df.columns = gls_df_cols
        cat_gls_df = pd.concat([cat_df, gls_df], axis=1)
        abs_pow_cols = [f'p{i_gls+1}' for i_gls in range(self.k_gls)]
        abs_pow_df = pd.DataFrame(abso_pow[0:i_combo, :])
        abs_pow_df.columns = abs_pow_cols
        worker_dataframe = pd.concat([cat_gls_df, abs_pow_df], axis=1)
        # At this point reorder the columns so that the power values are interleaved
        # with the catalog/glass columns
        cat_gls_pow_cols = []
        for i_gls in range(self.k_gls):
            cat_gls_pow_cols.extend([f'c{i_gls+1}', f'g{i_gls+1}', f'p{i_gls+1}'])
        worker_dataframe = worker_dataframe[cat_gls_pow_cols]
        # Add the sum of normalised powers
        worker_dataframe['f_1'] = sum_abs_norm_pow[0:i_combo]
        # Add the norm of the |CCP| * efl
        worker_dataframe['f_2'] = delta_color_F[0:i_combo]
        # Skip de Albuquerque F3, which is sum of some aberration residuals in aplanatic arrangement
        # Add the absolute thermal change in focus over the full delta_temp
        worker_dataframe['f_4'] = delta_temp_F[0:i_combo]
        worker_dataframe['f_5'] = delta_F[0:i_combo]
        # Any dataframe metadat/attributes
        worker_dataframe.attrs['combos_all_done'] = combos_all_done
        return worker_dataframe

    def dask_run_de_albuquerque(self, dask_client):
        '''
        Run the de Albuquerque method on a range of combinations produced by
        an iterator.

        Returns
        -------
        A pandas or dask dataframe with the results.
        '''  
        # A number of worker futures are produced
        self.dask_futures = []
        print('Launching workers...')
        for i_worker in range(self.dask_num_workers):
            # future = dask_client.submit(GlassCombo.dask_run_worker_old, self, i_worker)
            future = dask_client.submit(GlassCombo.dask_run_worker, self, i_worker)
            self.dask_futures.append(future)
        # Now wait for workers to complete and gather results
        result_dataframes = dask_client.gather(self.dask_futures)
        total_results = sum([len(result_df) for result_df in result_dataframes if result_df is not None])
        # Calculate and print a few run statistics
        self.last_run_total_results = total_results
        self.last_run_percent_max_results = 100.0 * total_results/self.max_result_rows
        print(f'Total number of result rows is {total_results}.')
        print(f'Percentage of maximum allowed results is {self.last_run_percent_max_results:.2f}%.')
        if total_results > 0:
            return pd.concat(result_dataframes, axis=0).reset_index(drop=True)
        else:
            return None

    def dask_produce_iterators(self, dask_client):
        '''
        Produce iterators for the du Albuquerque method on multiple dask.distributed workers.
        This must be called prior to running the problem using a dask_run method.

        This is also used to reset the iterators if a run is completed or if a partial run
        is to be aborted.

        Returns
        -------
        worker_iterators : list of iterators
            A list of iterators, one per anticipated dask worker process.
        '''
        # Create a number of iterators effectively equal to the number of workers
        worker_iterators = [None]*self.dask_num_workers
        max_combo_per_worker = [None]*self.dask_num_workers
        # First build dask_num_workers iterators for the lens group with maximum combos
        for i_worker in range(self.dask_num_workers):
            print(f'Producing Iterator for Worker {i_worker}')
            combo_index_range = self.comb_ind_worker_ranges[i_worker]  # 2-tuple of start end combo indices
            worker_iterator_max_combo = gen_combination(combo_index_range[0], combo_index_range[1], 
                                                range(self.num_gls_per_grp[self.i_grp_max_combo]),
                                                self.k_gls_per_grp[self.i_grp_max_combo])
            max_combo_per_worker[i_worker] = self.comb_ind_worker_splits[i_worker]  # gets multiplied by unsplit group combos
            # For each worker build a cartesian product iterator across lens groups
            grp_iterators = [None]*self.num_grp

            for i_grp in range(self.num_grp):
                print(f'Producing Iterator for Group # {i_grp}')                
                if i_grp == self.i_grp_max_combo:
                    grp_iterators[i_grp] = worker_iterator_max_combo
                else:
                    # This is an ordinary (unsplit) iterator over combinations for the lens group
                    max_combo_per_worker[i_worker] *= self.comb_per_grp[i_grp]
                    grp_iterators[i_grp] = itertools.combinations(range(self.num_gls_per_grp[i_grp]),
                                                    self.k_gls_per_grp[i_grp])
            worker_iterators[i_worker] = itertools.product(*grp_iterators)
        self.worker_iterators = worker_iterators
        self.max_combo_per_worker = max_combo_per_worker
        return worker_iterators

    @staticmethod
    def dask_client_shutdown(dask_client):
        '''
        Shuts down the dask client associated with this GlassCombo.
        '''
        dask_client.shutdown()

    def plot_combo_usingDataFrame(self, df, df_col_names, x_parm='vd', y_parm='nd'):
        '''
        Plot a diagram using a closed polygon for a set of glass combinations
        defined by stated columns in the rows of a dataframe. This dataframe would
        typically be the result of a de Albuquerque run.

        Parameters
        ----------
        df : pandas dataframe
            The dataframe should have at least two pairs of columns with glass catalog
            and glass names in the columns given in the 'df_cat_glt_cols' input.
        df_col_names : list of 2-tuples of str
            The names of the catalog and glass columns from which to extract the 
            data from the dataframe.
        x_parm : str
            The name of the x-axis parameter to plot. Default is 'vd', the abbe number.
        y_parm : str
            The name of the y-axis parameter to plot. Default is 'nd', the d-line
            refractive index.
        
        Returns
        -------

        '''
        from adjustText import adjust_text
        # Get a merged glass library
        gls_lib = copy.deepcopy(self.gls_lib_per_grp[0])
        for i_grp in range(1, self.num_grp):
            gls_lib.merge(self.gls_lib_per_grp[i_grp], inplace=True)
        # For each line in the dataframe plot a closed polygon
        all_x = []
        all_y = []
        all_gls = []
        for _, row in df.iterrows():
            x_vals = []
            y_vals = []
            for cat_col, gls_col in df_col_names:
                x_vals.append(gls_lib.library[row[cat_col]][row[gls_col]][x_parm])
                y_vals.append(gls_lib.library[row[cat_col]][row[gls_col]][y_parm])
                if row[gls_col] not in all_gls:
                    all_gls.append(row[gls_col])
                    all_x.append(gls_lib.library[row[cat_col]][row[gls_col]][x_parm])
                    all_y.append(gls_lib.library[row[cat_col]][row[gls_col]][y_parm])
            x_vals.append(x_vals[0])  # To plot a closed polygon
            y_vals.append(y_vals[0])
            plt.plot(x_vals, y_vals)
        # Do the labels
        plt.xlabel(x_parm)
        plt.ylabel(y_parm)
        labels = [plt.text(all_x[i], all_y[i], all_gls[i], ha='center', va='center') for i in range(len(all_x))]
        # Optimise positioning
        adjust_text(labels)

#=======================================
# End of GlassCombo class
#=======================================

# More utility and helper functions
def read_library(glassdir, catalog='all'):
    '''
    Get a list of all '*.agf' files in the directory, then call `parse_glassfile()` on each one.

    Parameters
    ----------
    glassdir : str
        The directory where we can find all of the *.agf files.
    catalog : str, optional
        If there is only one catalog of interest within the directory, then read only this one.

    Returns
    -------
    glass_library : dict
        A dictionary in which each entry is a glass catalog.

    Example
    -------
    >>> glasscat = read_zemax.read_glasscat('~/Zemax/Glasscat/')
    >>> nd = glasscat['schott']['N-BK7']['nd']
    '''

    glassdir = os.path.normpath(glassdir)
    files = glob.glob(os.path.join(glassdir, '*.[Aa][Gg][Ff]'))
    if (len(catalog) > 1) and isinstance(catalog, list):
        catalogs = [cat_name.lower() for cat_name in catalog] 
    else:
        catalogs = [catalog.lower()]

    ## Get the set of catalog names. These keys will initialize the glasscat dictionary.
    glass_library = {}
    cat_comment = {}
    cat_encoding = {}
    for f in files:
        this_catalog = os.path.basename(f)[:-4].lower()
        if (this_catalog.lower() not in catalogs) and (catalog != 'all'): continue
        (glass_library[this_catalog], cat_comment[this_catalog], 
            cat_encoding[this_catalog]) = parse_glass_file(f)

    return(glass_library, cat_comment, cat_encoding)

## =============================================================================
from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE

# Define some Byte Order Markers (BOM) for Unicode encoded files
# The BOM is a magic number appearing at the start of the file, U+FEFF, which indicates the
# byte endian-ness. Most UTF-8 (ASCII compatible) encoded files do not have the BOM marker
# and check_bom will not return anything, in which it will be assumed that the data is
# 'latin-1' encoded.
BOMS = (
    (BOM_UTF8, "UTF-8"),
    (BOM_UTF16_BE, "UTF-16-BE"),
    (BOM_UTF16_LE, "UTF-16-LE"),
    (BOM_UTF32_BE, "UTF-32-BE"),
    (BOM_UTF32_LE, "UTF-32-LE"),
)

def check_bom(data):
    '''
    Crude method of checking for the presence of the BOM magic number at the start of a data
    fragment.
    '''
    return [encoding for bom, encoding in BOMS if data.startswith(bom)]

def parse_glass_file(filename):
    '''
    Read a Zemax glass file (*.agf') and return its contents as a Python dictionary.

    Parameters
    ----------
    filename : str
        The file to parse.

    Returns
    -------
    glass_catalog : dict
        The dictionary containing glass data for all glasses in the file.
    '''
    # First try to guess the file encoding
    with open(filename, 'rb') as f:  # First open in binary mode
        data = f.read(20)  # Read a scrap of data from the start of the file
    encoding_guesses = check_bom(data)
    if encoding_guesses:
        encoding_guess = encoding_guesses[0]
    else:
        encoding_guess = 'latin-1'
    with open(filename, 'r', encoding=encoding_guess) as cat_file:
        cat_data = cat_file.readlines()  # read the whole file as a list of strings
    cat_comment = ''  # A comment pertaining to the whole catalog file
    glass_catalog = {}
    glassname = ''
    # print(f'Reading Catalog {filename}')
    for i_line, line in enumerate(cat_data):
        if not line.strip(): continue  # Blank line
        if line.startswith('CC '):
            cat_comment = line[2:].strip()
            continue
        if line.startswith('NM '):  # Glass name, dispersion formula type, n_d v_d, status, melt frequency
            nm = line.split()
            glassname = nm[1]
            # print(f'Reading Glass {glassname}')
            glass_catalog[glassname] = {}
            glass_catalog[glassname]['text'] = ''  # Next glass data starts with this
            glass_catalog[glassname]['dispform'] = int(float(nm[2]))
            glass_catalog[glassname]['nd'] = float(nm[4])
            glass_catalog[glassname]['vd'] = float(nm[5])
            glass_catalog[glassname]['exclude_sub'] = 0 if (len(nm) < 7) else int(float(nm[6]))
            glass_catalog[glassname]['status'] = 5 if (len(nm) < 8) else int(float(nm[7]))
            status = glass_catalog[glassname]['status']
            if status < 0 or status > 5: status = 5  # Unknown status
            glass_catalog[glassname]['stat_txt'] = ['Standard', 'Preferred', 'Obsolete', 'Special', 'Melt', 'Unknown'][glass_catalog[glassname]['status']]
            glass_catalog[glassname]['meltfreq'] = 0 if ((len(nm) < 9) or (nm.count('-') > 0)) else int(float(nm[8]))
        elif line.startswith('GC '):  # Individual glass comment
            glass_catalog[glassname]['comment'] = line[2:].strip() 
        elif line.startswith('ED '):  # Thermal expansion data (TCE), density, relative partial dispersion
            ed = line.split()
            glass_catalog[glassname]['tce'] = float(ed[1])
            glass_catalog[glassname]['density'] = float(ed[3])
            glass_catalog[glassname]['dpgf'] = float(ed[4])
            glass_catalog[glassname]['ignore_thermal_exp'] = 0 if (len(ed) < 6) else int(float(ed[5]))
        elif line.startswith('CD '):  # Dispersion formula coefficients
            cd = line.split()[1:]
            # Check the number of coefficients
            n_coeff = num_coeff[glass_catalog[glassname]['dispform']-1]
            # Use additional lines if there is insufficient number of coefficients found
            lookahead = 1
            while len(cd) < n_coeff:
                cd.extend(cat_data[i_line + lookahead].split())  # Extend list to include next line
                lookahead += 1
            glass_catalog[glassname]['cd'] = [float(a) for a in cd]
        elif line.startswith('TD '):  # dn/dT formula data
            td = line.split()[1:]
            if not td: continue     ## the Schott catalog sometimes uses an empty line for the "TD" label
            lookahead = 1
            # Some catalogs split this data over several lines, so do a lookahead if there is not enough data
            while len(td) < 7:  # There should be seven numbers D0 D1 D2 E0 E1 lamda_tk ref_temp
                td.extend(cat_data[i_line + lookahead].split())  # Extend list to include next line
                lookahead += 1
            glass_catalog[glassname]['td'] = [float(a) for a in td]
        elif line.startswith('OD '):  # Relative cost and environmental data
            od = line.split()[1:]
            od = string_list_to_float_list(od)
            glass_catalog[glassname]['relcost'] = od[0]
            glass_catalog[glassname]['cr'] = od[1]
            glass_catalog[glassname]['fr'] = od[2]
            glass_catalog[glassname]['sr'] = od[3]
            glass_catalog[glassname]['ar'] = od[4]
            if (len(od) == 6):
                glass_catalog[glassname]['pr'] = od[5]
            else:
                glass_catalog[glassname]['pr'] = -1.0
        elif line.startswith('LD '):  # Valid range for dispersion data
            ld = line.split()[1:]
            glass_catalog[glassname]['ld'] = [float(a) for a in ld]
        elif line.startswith('IT '):  # Transmission data
            it = line.split()[1:]
            it_row = [float(a) for a in it]
            if ('it' not in glass_catalog[glassname]):
                glass_catalog[glassname]['it'] = {}
                glass_catalog[glassname]['it']['wavelength'] = []
                glass_catalog[glassname]['it']['transmission'] = []
                glass_catalog[glassname]['it']['thickness'] = []
            glass_catalog[glassname]['it']['wavelength'].append(it_row[0])
            if len(it_row) > 1:
                glass_catalog[glassname]['it']['transmission'].append(it_row[1])
            else:
                glass_catalog[glassname]['it']['transmission'].append(np.NaN)

            if len(it_row) > 2:
                glass_catalog[glassname]['it']['thickness'].append(it_row[2])
            else:
                glass_catalog[glassname]['it']['thickness'].append(np.NaN)
            # Create them as numpy arrays as well
            glass_catalog[glassname]['it']['wavelength_np'] = np.array(glass_catalog[glassname]['it']['wavelength'])
            glass_catalog[glassname]['it']['transmission_np'] = np.array(glass_catalog[glassname]['it']['transmission'])
            glass_catalog[glassname]['it']['thickness_np'] = np.array(glass_catalog[glassname]['it']['thickness'])
        if glassname:
            glass_catalog[glassname]['text'] += line

    # f.close()
    if glassname:  # Strongly suggests file was read with correctly guessed encoding
        cat_encoding = encoding_guess
    else:
        cat_encoding = ''
    return(glass_catalog, cat_comment, cat_encoding)

## =================================================================================================
def string_list_to_float_list(x):
    '''
    Convert a list of strings to a list of floats, where a string value of '-' is mapped to a
    floating point value of -1.0, and an empty input list produces a length-10 list of -1.0's.

    Parameters
    ----------
    x : list
        The list of strings to convert

    Returns
    -------
    res : list of floats
        The converted results.
    '''
    npts = len(x)
    if (npts == 0) or ((npts == 1) and (x[0].strip() == '-')):
        return([-1.0] * 10)

    res = []
    for a in x:
        if (a.strip() == '-'):
            res.append(-1.0)
        else:
            try:
                res.append(float(a))
            except:
                res.append(np.NaN)

    return(res)

## =================================================================================================
def find_catalog_for_glassname(glass_library, glassname):
    '''
    Search for the catalog containing a given glass.

    Note that this is not a perfect solution --- it is common for multiple catalogs to share glass
    names, and this function will only return the first one it finds.

    Parameters
    ----------
    glass_library : ZemaxGlassLibrary
        The glass library to search through.
    glassname : str
        The name of the glass to search for.

    Returns
    -------
    catalog : str
        The name of the catalog where the glass is found. If not found, then return None.
    '''
    for catalog in glass_library:
        if glassname in glass_library[catalog]:
            return(catalog)
    return(None)

## =================================================================================================
def polyeval_Horner(x, poly_coeffs):
    '''
    Use Horner's rule for polynomial evaluation.

    Assume a polynomial of the form \
        p = c[0] + (c[1] * x) + (c[2] * x**2) + (c[3] * x**3) + ... + (c[N] * x**N).

    Parameters
    ----------
    x : array_like
        The abscissa at which to evaluate the polynomial.
    poly_coeffs : array_like
        The vector of polynomial coefficients.

    Returns
    -------
    p : ndarray
        The polynomial evaluated at the points given in x.
    '''

    ncoeffs = np.alen(poly_coeffs)
    p = np.zeros(np.alen(x))
    for n in np.arange(ncoeffs-1,-1,-1):
        p = poly_coeffs[n] + (x * p)
        #print('n=%i, c=%f' % (n, coeffs[n]))
    return(p)

## =================================================================================================
def get_colors(num_colors):
    '''
    Make a list of 16 discernably different colors that can be used for drawing plots.

    Returns
    -------
    mycolors : list of floats
        A 16x4 list of colors, with each color being a 4-vector (R,G,B,A).
    '''

    mycolors = [None]*16
    mycolors[0]  = [0.0,0.0,0.0,1.0]        ## black
    mycolors[1]  = [1.0,0.0,0.0,1.0]        ## red
    mycolors[2]  = [0.0,0.0,1.0,1.0]        ## blue
    mycolors[3]  = [0.0,0.5,0.0,1.0]        ## dark green
    mycolors[4]  = [1.0,0.5,0.0,1.0]        ## orange
    mycolors[5]  = [0.0,0.5,0.5,1.0]        ## teal
    mycolors[6]  = [1.0,0.0,1.0,1.0]        ## magenta
    mycolors[7]  = [0.0,1.0,0.0,1.0]        ## lime green
    mycolors[8]  = [0.5,0.5,0.0,1.0]        ## olive green
    mycolors[9]  = [1.0,1.0,0.0,1.0]        ## yellow
    mycolors[10] = [0.5,0.0,0.0,1.0]        ## maroon
    mycolors[11] = [0.5,0.0,0.5,1.0]        ## purple
    mycolors[12] = [0.7,0.7,0.7,1.0]        ## bright grey
    mycolors[13] = [0.0,1.0,1.0,1.0]        ## aqua
    mycolors[14] = [0.4,0.4,0.4,1.0]        ## dark grey
    mycolors[15] = [0.0,0.0,0.5,1.0]        ## navy blue
    return(mycolors[:num_colors])

## =============================================================================================
def interp1d(x_old, y_old, x_new, **kwargs):
    '''
    A simple wrapper around the scipy `interp1d`, requiring only one function call rather than two,
    and also allowing for `x_old` to be monotonic in either direction and not just monotonic
    increasing.

    Parameters
    ----------
    x_old: ndarray
        The vector of abscissa values in the input data.
    y_old : ndarray
        The vector of ordinate values in the input data.
    x_new : ndarray
        The vector of desired evaluation points in the interpolated output.

    Returns
    -------
    y_new : ndarray
        The vector of interpolated points (evaluated at sampling points x_new).
    '''

    import scipy.interpolate
    reversed = (x_old[0] > x_old[-1])
    if reversed:
        x = np.array(x_old[::-1])
        y = np.array(y_old[::-1])
    else:
        x = np.array(x_old)
        y = np.array(y_old)

    ## If the raw data does not support the full desired x-range, then extrapolate the ends of the data.
    if (np.amin(x) > np.amin(x_new)):
        x = np.append(np.amin(x_new), x)
        y = np.append(y[0], y)
    if (np.amax(x) < np.amax(x_new)):
        x = np.append(x, np.amax(x_new))
        y = np.append(y, y[-1])

    if ('fill_value' in kwargs):
        del kwargs['fill_value']
    if ('bounds_error' in kwargs):
        del kwargs['bounds_error']

    func = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=None, **kwargs)
    y_new = func(x_new)

    return(y_new)

## =============================================================================================
if (__name__ == '__main__'):

    glasslib = ZemaxGlassLibrary(catalog='schott', wavemin=400.0, wavemax=700.0, nwaves=100)

    print('Number of glasses found in the library: ' + str(glasslib.nglasses))
    print('Glass catalogs found:')
    print(glasslib.catalogs)
    print('Glass names found:')
    print(glasslib.glasses)

    ## Demonstrate the ability to plot dispersion curves for any glass.
    glasslib.plot_dispersion('N-BK7', 'schott')
    glasslib.plot_dispersion('SF66', 'schott', polyfit=True, fiterror=True)

    ## Demonstrate the ability to plot the temperature dependence of index.
    temperatures = (0,100,200,300,400)
    glasslib.plot_temperature_dependence('N-BK7', 'schott', 550.0, temperatures)

    ## Demonstrate the ability to plot curves for *any* glass property.
    print('Now analyzing ALL of the glass catalogs ...')
    glasslib = ZemaxGlassLibrary(catalog='all')
    print('Number of glasses found in the library: ' + str(glasslib.nglasses))
    print('Glass catalogs found:')
    print(glasslib.catalogs)
    # glasslib.plot_catalog_property_diagram('all', prop1='vd', prop2='nd')
    # glasslib.plot_catalog_property_diagram('all', prop1='nd', prop2='dispform')
    glasslib.plot_catalog_property_diagram('schott', prop1='n0', prop2='n1')
    # glasslib.plot_catalog_property_diagram('all', prop1='n0', prop2='n1')
    #glasslib.plot_catalog_property_diagram('cdgm', prop1='vd', prop2='nd')

    ## Demonstrate how to pretty-print glass data.
    # glasslib.pprint('schott')          ## print all the glass info found in the Schott glass catalog
    # glasslib.pprint()                  ## print all of the glass info for the entire library of glasses
    glasslib.pprint('schott','SF66')   ## print the info for SF66 glass in the Schott glass catalog

    ## Now show something in the infrared.
    print('Now analyzing the "Infrared" glass catalog ...')
    glasslib = ZemaxGlassLibrary(degree=5, wavemin=7500.0, wavemax=12000.0, catalog='infrared')
    print('Number of glasses found in the library: ' + str(glasslib.nglasses))
    print('Glass catalogs found:', glasslib.catalogs)
    glasslib.plot_dispersion('ZNS_BROAD', 'infrared')

    #glasslib = ZemaxGlassLibrary(wavemin=7000.0, wavemax=12000.0, catalog='temp')
    #glasslib.plot_catalog_property_diagram('temp', prop1='n0', prop2='n1')

    plt.show()
