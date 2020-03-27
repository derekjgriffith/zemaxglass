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
try:
    from IPython.display import clear_output
except ImportError as e:
    clear_output_possible = False
import pandas as pd
import re


"""
This file contains a set of utilities for reading Zemax glass (*.agf) files, analyzing glass
properties, and displaying glass data.

See LICENSE.txt for a description of the MIT/X license for this file.
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

def zemax_dispersion_formula(wv, dispform, coefficients):
    """
    Calculate material refractive indices according to the various dispersion formulae defined in the Zemax manual.
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
        1 : Schott
        2 : Sellmeier 1
        3 : Herzberger
        4 : Sellmeier 2
        5 : Conrady
        6 : Sellmeier 3
        7 : Handbook of Optics 1
        8 : Handbook of Optics 2
        9 : Sellmeier 4
        10: Extended 1
        11: Sellmeier 5
        12: Extended 2
        13: Extended 3
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

    def __init__(self, dir=None, wavemin=400.0, wavemax=700.0, nwaves=300, catalog='all', glass_match=None,
                sampling_domain='wavelength', degree=3, discard_off_band=False, debug=False):
        '''
        Initialize the glass library object.

        Parameters
        ----------
        wavemin : float, optional
            The shortest wavelength (nm) in the spectral region of interest.
        wavemax : float, optional
            The longest wavelength (nm) in the spectral region of interest.
        nwaves : float, optional
            The number of wavelength samples to use.
        catalog : str
            The catalog or list of catalogs to look for in "dir".
        glass_match : str
            Regular expression to match. The glass is only included in the returned instance if the glass name
            matches this regular expression. Default is None - all glasses in catalog are returned.     
        sampling_domain : str, {'wavelength','wavenumber'}
            Whether to sample the spectrum evenly in wavelength or wavenumber.
        degree : int, optional
            The polynomial degree to use for fitting the dispersion spectrum.
        discard_off_band : boolean
            If set True, will discard glasses where the valid spectral range does not fully cover
            the interval wavemin to wavemax.
        '''

        self.debug = debug
        self.degree = degree                    ## the degree of polynomial to use when fitting dispersion data
        #self.basis = basis                     ## the type of basis to use for polynomial fitting ('Taylor','Legendre')
        self.sampling_domain = sampling_domain  ## the domain ('wavelength' or 'wavenumber') in which to evenly sample the data

        if (dir == None):
            dir = os.path.dirname(os.path.abspath(__file__)) + '/AGF_files/'

        self.dir = dir
        self.library, self.cat_comment, self.cat_encoding = read_library(dir, catalog=catalog)
        # Remove glasses where requested wavelength interval is not covered by the valid interval of the dipersion data
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
        # Discard glasses that do not match the regular expression
        if glass_match is not None:
            cat_discard_list = []
            for catalogue in self.library.keys():
                discard_list = []
                for glass in self.library[catalogue].keys():
                    if not re.match(glass_match, glass):
                        # print(f'Discarding {catalogue.capitalize()} {glass} RE mismatch')
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

        self.pressure_ref = 1.0113e5   ## the dispersion measurement reference pressure, in Pascals
        self.temp_ref = 20.0           ## the dispersion measurement reference temperature, in degC

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
    def __getattr__(self, name):
        '''
        Redirect the default __getattr__() function so that any attempt to generate a currently nonexisting attribute
        will trigger a method to generate that attribute from existing attributes.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.
        '''

        if (name == 'nglasses'):
            nglasses = 0
            for catalog in self.library:
                for glass in self.library[catalog]:
                    nglasses += 1
            return(nglasses)
        elif (name == 'catalogs'):
            catalogs = self.library.keys()
            return(catalogs)
        elif (name == 'glasses'):
            glasses = []
            for catalog in self.library:
                glasses.extend(self.library[catalog].keys())
            return(glasses)

        return

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

        for catalog in self.library:
            if (catalog not in catalogs): continue
            print(catalog.capitalize() + ':')
            for glassname in self.library[catalog]:
                if (glass != None) and (glassname != glass.upper()): continue
                glassdict = self.library[catalog][glassname]
                print('  ' + glassname + ':')
                print('    nd       = ' + str(glassdict['nd']))
                print('    vd       = ' + str(glassdict['vd']))
                print('    dispform = ' + str(glassdict['dispform']) + 
                      ' the ' + ZemaxGlassLibrary.dispformulas[glassdict['dispform']] + ' formula.')
                if ('tce' in glassdict):  # thermal coefficient of expansion
                    print('    tce      = ' + str(glassdict['tce']))
                if ('density' in glassdict):  # density in g/cc ?
                    print('    density  = ' + str(glassdict['density']))
                if ('dpgf' in glassdict):  # relative partial dispersion
                    print('    dpgf     = ' + str(glassdict['dpgf']))
                if ('cd' in glassdict):  # dispersion formula coefficients
                    print('    cd       = ' + str(glassdict['cd']))
                if ('td' in glassdict):  # thermal data
                    print('    td       = ' + str(glassdict['td']))
                if ('od' in glassdict):  # environmental data
                    print('    od       = ' + str(glassdict['od']))
                if ('ld' in glassdict):  # valid range of dispersion relation
                    print('    ld       = ' + str(glassdict['ld']))
                if ('interp_coeffs' in glassdict):  # interpolation coefficients for polynomial fit
                    print('    coeffs   = ' + repr(glassdict['interp_coeffs']))

        print('')
        return

    def asDataFrame(self, fields, catalog=None, glass=None):
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
                'meltfreq' : Melt frequency of the glass
                'comment' : string comment found in the catalog file
                'relcost' : relative cost of the glass to N-BK7/S-BSL7
                'cr' : Various environmental ratings 
                'fr' : 
                'sr' :
                'ar' : Acid resistance rating
                'pr' : Phosphate resistance rating
            Other fields that have been added to the glass instance should also work.

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
    def simplify_schott_catalog(self, zealous=False):
        '''
        Remove redundant, little-used, and unusual glasses from the Schott glass catalog.

        Parameters
        ----------
        zealous : bool, optional
            Whether to remove the "high transmission" and close-to-redundant glasses.
        '''

        if ('schott' not in self.library):
            return

        schott_glasses = []

        for glass in self.library['schott']:
            schott_glasses.append(glass)

        ## Remove the "inquiry glasses".
        I_glasses = ['FK3', 'N-SK10', 'N-SK15', 'BAFN6', 'N-BAF3', 'N-LAF3', 'SFL57', 'SFL6', 'SF11', 'N-SF19', 'N-PSK53', 'N-SF64', 'N-SF56', 'LASF35']
        num_i = len(I_glasses)

        ## Remove the "high-transmission" duplications of regular glasses.
        H_glasses = ['LF5HT', 'BK7HT', 'LLF1HT', 'N-SF57HT', 'SF57HT', 'LF6HT', 'N-SF6HT', 'F14HT', 'LLF6HT', 'SF57HHT', 'F2HT', 'K5HT', 'SF6HT', 'F8HT', 'K7HT']
        num_h = len(H_glasses)

        ## Remove the "soon-to-be-inquiry" glasses from the Schott catalog.
        N_glasses = ['KZFSN5', 'P-PK53', 'N-LAF36', 'UBK7', 'N-BK7']
        num_n = len(N_glasses)

        ## Remove the Zinc-sulfide and zinc selenide glasses.
        ZN_glasses = ['CLEARTRAN_OLD', 'ZNS_VIS']
        num_zn = len(ZN_glasses)

        ## "zealous": remove the "P" glasses specifically designed for hot press molding, and several glasses that are nearly identical to others in the catalog.
        Z_glasses = ['N-F2', 'N-LAF7', 'N-SF1', 'N-SF10', 'N-SF2', 'N-SF4', 'N-SF5', 'N-SF57', 'N-SF6', 'N-ZK7', 'P-LASF50', 'P-LASF51', 'P-SF8', 'P-SK58A', 'P-SK60']
        num_z = len(Z_glasses)

        for glass in schott_glasses:
            remove = (glass in I_glasses) or (glass in H_glasses) or (glass in N_glasses) or (glass in ZN_glasses)
            if zealous:
                remove = remove or (glass in Z_glasses)
            if remove:
                del self.library['schott'][glass]

        ## Refresh any existing information in the library.
        if hasattr(self, 'nglasses'):
            nglasses = 0
            for catalog in self.library:
                for glass in self.library[catalog]:
                    nglasses += 1
            self.nglasses = nglasses
        elif (name == 'glasses'):
            glasses = []
            for catalog in self.library:
                glasses.extend(self.library[catalog].keys())
            self.glasses = glasses

        return

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
        ## This is the formula used in Zemax, but over what wavelength region is this formula valid?
        ## Reference : F. Kohlrausch, Praktische Physik, 1968, Vol 1, page 408
        if (glass.upper() == 'AIR'):
            T_ref = 20.0
            P_ref = self.pressure_ref   ## the dispersion measurement reference pressure in Pascals
            n_ref = 1.0 + ((6432.8 + ((2949810.0 * w**2) / (146.0 * w**2 - 1.0)) + ((25540.0 * w**2) / (41.0 * w**2 - 1.0))) * 1.0e-8)
            indices = 1.0 + ((n_ref - 1.0) / (1.0 + (T_ref - 15.0) * 3.4785e-3)) * (P / P_ref)
        if (glass.upper() == 'VACUUM'):
            indices = np.ones_like(w)

        if (dispform == 0):
            ## use this for AIR and VACUUM
            pass
        else:
            indices = zemax_dispersion_formula(w, dispform, cd)

        ## If 'TD' is included in the glass data, then include pressure and temperature dependence of the lens
        ## environment. From Schott's technical report "TIE-19: Temperature Coefficient of the Refractive Index".
        ## The above "indices" data are assumed to be from the reference temperature T_ref. Now we add a small change
        ## delta_n to it due to a change in temperature.
        if ('td' in self.library[catalog][glass]):
            td = self.library[catalog][glass]['td']
            dT = T - T_ref
            dn = ((indices**2 - 1.0) / (2.0 * indices)) * (td[0] * dT + td[1] * dT**2 + td[2] * dT**3 + ((td[3] * dT + td[4] * dT**2) / (w**2 - td[5]**2)))
            indices = indices + dn

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

    def get_indices(self, wv=wv_d, catalog=None, glass=None):
        '''
        Get the refractive indices of a glass for a specified set of wavelngths.

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
                # Calculate the refractive index
                glass_indices = zemax_dispersion_formula(wv, self.library[this_catalog][this_glass]['dispform'],
                                                             self.library[this_catalog][this_glass]['cd'])
                if indices.size > 0:
                    indices = np.vstack((indices, glass_indices))
                else:
                    indices = glass_indices
        
        return catalog_list, glass_list, indices

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
        wv_centre : float, optional
            The centre wavelength to use for the generalised Abbe number calculation.
            Default is the yellow Helium d line at 587.5618 nm.
            If wv_centre is an array, then all the wavelength inputs must also be arrays of the same length.
            In this case, the color correction merit is the product of the merit values for the different
            wavelength regions effectively so specified.           
        wv_x : float, optional
            The numerator lower wavelength to use for the generalised partial dispersion calculation.
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
            color correction.
        
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
            List of field names to extracted from the glass catalog data
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
            List of glass names
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
## End of ZemaxLibrary class
## =============================================================================

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
        catalogs = catalog
    else:
        catalogs = [catalog]

    ## Get the set of catalog names. These keys will initialize the glasscat dictionary.
    glass_library = {}
    cat_comment = {}
    cat_encoding = {}

    for f in files:
        # print('Reading ' + f + ' ...')
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
    # print(f'Opening Catalog {filename}')
    # First try to guess the file encoding
    with open(filename, 'rb') as f:  # First open in binary mode
        data = f.read(20)  # Read a scrap of data from the start of the file
    encoding_guesses = check_bom(data)
    if encoding_guesses:
        encoding_guess = encoding_guesses[0]
    else:
        encoding_guess = 'latin-1'
    # print(f'Encoding Guess {encoding_guess}')
    f = open(filename, 'r', encoding=encoding_guess)
    cat_comment = ''  # A comment pertaining to the whole catalog file
    glass_catalog = {}
    glassname = ''
    # print(f'Reading Catalog {filename}')
    for line in f:
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
            glass_catalog[glassname]['status'] = 0 if (len(nm) < 8) else int(float(nm[7]))
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
            glass_catalog[glassname]['cd'] = [float(a) for a in cd]
        elif line.startswith('TD '):  # dn/dT formula data
            td = line.split()[1:]
            if not td: continue     ## the Schott catalog sometimes uses an empty line for the "TD" label
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

    f.close()
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
