---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Motivation
The opto-thermal coefficient of a glass material quantifies the amount by which the optical power (or focal length) of a lens made from the material will change with temperature. It takes into account both the change in refractive index of the material, as well as the change in dimensions of the lens with a change in temperature.

Optical system comprising multiple lenses in multiple materials can be brought closer to athermalisation by choosing materials with opto-thermal coefficients that tend to bring the sum of the temperature-induced changes closer to zero. This is made easier by the fact that some materials have negative opto-thermal coeffients and some have positive coefficients.

This notebook computes a table of opto-thermal coefficients for catalogue glasses where the relevant information (as required by the formulae and procedures presented below) is provided by the manufacturer. Glasses from Schott, Ohara and [CDGM](http://www.cdgm.eu/CDGM/CDGM.html) are included in the tables.


# Formulae and Procedure
The opto-thermal coefficient of a glass material is defined (Reshidko 2013) as
$$\gamma = \frac{\frac{\mathrm{d}n}{\mathrm{d}T}}{n-1}-\alpha, $$
where $\frac{\mathrm{d}n}{\mathrm{d}T}$ is the thermal coefficient of refractive index (units $K^{-1}$), $n$ is the nominal refractive index (at some reference wavelength, temperature and pressure) and $\alpha$ is the linear coefficient of thermal expansion (CTE) of the glass (in $m/m/K$ which is just $K^{-1}$).
The change in focal length $\Delta f$ of a lens of nominal focal length $f$ is related to the opto-thermal coefficient as
$$\frac{\Delta f}{f} = -\gamma \cdot \Delta T.$$

The computation is performed using a model for the change in refractive index (from nominal) for the change in temperature $\Delta T$. The absolute change in refractive index is
$$\Delta n_{abs} = \frac{n^2-1}{2n}\left[ D_0 \Delta T + D_1 \Delta T^2 + D_2 \Delta T^3 + \frac{E_0 \Delta T + E_1 \Delta T^2}{\lambda^2 - \lambda^2_{tk}}\right],$$
where the constants $D_0$, $D_1$, $D_2$, $E_0$, $E_1$ and $\lambda_{tk}$ are provided in the material datasheet (for most standard catalogue glasses). The relevant data for fused silica is taken from the data for Schott LITHOSIL-Q, although Schott no longer offers this product.

In the case of certain materials, very limited data is available, for example a single $\frac{\mathrm{d}n}{\mathrm{d}T}$ value. Here the only option is to assume that only the $D_0$ value is non-zero, resulting in the relationship
$$\Delta n_{abs}=\frac{n^2-1}{2n}D_0 \Delta T,$$
and the value of $D_0$ to be used would be
$$D_0=\frac{2n}{n^2-1} \frac{\mathrm{d}n}{\mathrm{d}T}$$

In this case, the mean opto-thermal coefficient for a specific temperature range is required. Suppose the temperature limits of interest are $T_{lo}$ and $T_{hi}$ and $\Delta T = T_{hi} - T_{lo}$. 
The reference temperature for the catalogue refractive indices is $T_0=20^\circ$C and the reference pressure is $P=1$ atmosphere. The catalogue dispersion formula always provides the indices of the material relative to air at this temperature and pressure.

The procedure used in Zemax to compute the relative refractive index of a glass at a specific wavelength $\lambda$ is as follows (taken from the Zemax manual):

1. Scale the wavelength in question to be in air at the reference (catalogue) temperature of the glass. This notebook generally ignores this small wavelength shift between vacuum and 1 atmosphere. Effectively, all wavelengths are assumed to be provided in air at catalogue reference temperature and pressure. 
2. Compute the relative index of refraction of the glass from the catalogue dispersion formulas.
3. Compute the index of air at the reference (catalogue) temperature of the glass using the Kohlrausch formula below.
4. Compute the absolute index of the glass (relative to vacuum) at the reference temperature of the glass by multiplying the catalogue relative index by the index of air computed in the previous step.
5. Compute the change in absolute index of refraction of the glass at the specified temperature (using the formula for $\Delta n_{abs}$ given above).
6. Compute the index of air at the system temperature and pressure, again using the Kohlrausch formula given below.
7. Compute the index of the glass relative to the air at the system temperature and pressure by dividing by the index of air computed in the previous step.

The refractive index of air is computed (F. Kohlrausch, Praktische Physik, 1968, Vol 1, page 408) as
$$n_{air}=1+\frac{\left( n_{ref}-1\right)P}{1+(T-15)\cdot(3.4785\times10^{-3})},$$
where
$$n_{ref} = 1+\left[ 6432.8+\frac{2949810 \lambda^2}{146\lambda^2 - 1} + \frac{25540\lambda^2}{41\lambda^2-1} \right] \cdot 1.0\times10^{-8},$$
where $T$ is the temperature in Celsius, $P$ is the relative air pressure (dimensionless) and $\lambda$ has units of $\mu m$.

## Refractivity of Air

While not really relevant to the spectral region considered here, the Kohlrausch formula for the refractive index of air may only be valid within a restricted wavelength region. Alternative formulas for the refractive index of air include the Edlén and Ciddor equations. These can be computed using the [ref_index](https://pypi.org/project/ref_index/) package. NIST indicates that the [Edlén](https://emtoolbox.nist.gov/Wavelength/Edlen.asp) and [Ciddor](https://emtoolbox.nist.gov/Wavelength/Ciddor.asp) equations are valid over the spectral range 300 nm to 1700 nm. 

<!-- #endregion -->

```python
# General imports
import numpy as np
import ZemaxGlass
import re
import os
import pandas as pd
%load_ext autoreload
%autoreload 1
%aimport ZemaxGlass
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.core.display import Math, Markdown, Latex
pd.set_option('display.max_rows', 1000)
```

```python
latex_flag = True  # Set this True when downloading .tex for reporting purposes
# Choose wavelength and temperature ranges
wv_lo = 450.0  # nm
wv_hi = 850.0  # nm
temp_lo = -10.0  # deg C
temp_hi = 40.0  # deg C
display(Latex(f'Opto-thermal coefficients computed for temperature range {temp_lo}$^\circ$C to {temp_hi}$^\circ$C'))
display(Latex(f'Reference wavelength is Sodium d-line at 586.9 nm'))
```

```python
# Read the current Ohara catalog, filtering for only those glasses that start with 'S-' that are standard or preferred status
# Read the Ohara catalog
ohara_catalog = 'ohara_20200623'
ohara = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, 
                                     catalog=ohara_catalog, glass_match='S-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
print(ohara.library.keys())
# Compute opto-thermal coefficients
ohara.add_opto_thermal_coeff(temp_lo=temp_lo, temp_hi=temp_hi)
```

```python
def gls_name_formatter(gls):
    return 
def dndT_formatter(dndT):
    return '%1.3f' % (dndT * 1.0e6)
def opto_therm_coeff_formatter(gamma):
    return '%1.3f' % (gamma * 1.0e6)
def n_d_formatter(n_d):
    return '%10.4f' % n_d
def nu_d_formatter(nu_d):
    return '%10.2f' % nu_d

ohara_df = ohara.asDataFrame(fields=['nd', 'vd', 'dndT', 'opto_therm_coeff']).sort_values(['nd', 'vd'], ascending=[1, 0])
# Replace catalog name with just Ohara
ohara_df.replace(to_replace=ohara_catalog, value='Ohara', inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Table 1. Ohara Opto-thermal Coefficients Sorted by Refractive Index'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(ohara_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))
    ohara_df.sort_values(by='gls', inplace=True)
    display(Latex('\\clearpage\\begin{center}Table 2. Ohara Opto-thermal Coefficients Sorted by Glass Name'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(ohara_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))    
    display(Latex('\\clearpage'))
    
else:
    display(ohara_df)
```

```python
# Read the Schott catalog, only N- type glasses that are standard or preferred
schott_catalog = 'schott_20180601'
schott = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, 
                                     catalog=schott_catalog, glass_match='N-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
print(schott.library.keys())
schott.add_opto_thermal_coeff(temp_lo=temp_lo, temp_hi=temp_hi)
```

```python
# Create a pandas dataframe 
schott_df = schott.asDataFrame(fields=['nd', 'vd', 'dndT', 'opto_therm_coeff']).sort_values(['nd', 'vd'], ascending=[1, 0])
# Replace catalog name with just Ohara
schott_df.replace(to_replace=schott_catalog, value='Schott', inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Table 3. Schott Opto-thermal Coefficients Sorted by Refractive Index'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(schott_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))
    schott_df.sort_values(by='gls', inplace=True)
    display(Latex('\\clearpage\\begin{center}Table 4. Schott Opto-thermal Coefficients Sorted by Glass Name'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(schott_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))
    display(Latex('\\clearpage'))
else:
    display(schott_df)
```

```python
# Read Schott LITHOSIL-Q
# Read the Schott catalog, only N- type glasses that are standard or preferred
schott_sil = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, 
                                     catalog=schott_catalog, glass_match='LITHOSIL',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10)
schott_sil.add_opto_thermal_coeff(temp_lo=temp_lo, temp_hi=temp_hi)
```

```python
# Schott LITHOSIL-Q
# This material is obsolete, but it has catalogue values for the behaviour of fused silica with temperature

schott_sil_df = schott_sil.asDataFrame(fields=['nd', 'vd', 'dndT', 'opto_therm_coeff']).sort_values(['nd', 'vd'], 
                                                                                                    ascending=[1, 0])
schott_sil_df.replace(to_replace=schott_catalog, value='Schott', inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Table 5. Opto-thermal Coefficients for Schott LITHOSIL-Q'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(schott_sil_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Material','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))

    display(Latex('\\clearpage'))
else:
    display(schott_sil_df)
```

```python
print(f"LITHOSIL-Q dn/dT : {schott_sil.library['schott_20180601']['LITHOSIL-Q']['dndT']} per K")
print(f"LITHOSIL-Q gamma : {schott_sil.library['schott_20180601']['LITHOSIL-Q']['opto_therm_coeff']} per K")
```

```python
# Read the CDGM catalogue. Only standard and preferred status materials
cdgm_catalog = 'cdgm_201904'
cdgm = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, glass_match='H-',
                                     catalog=cdgm_catalog,
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
print(cdgm.library.keys())
cdgm.add_opto_thermal_coeff(temp_lo=temp_lo, temp_hi=temp_hi)
```

```python
# Create a pandas dataframe 
cdgm_df = cdgm.asDataFrame(fields=['nd', 'vd', 'dndT', 'opto_therm_coeff']).sort_values(['nd', 'vd'], ascending=[1, 0])
# Replace catalog name with just Ohara
cdgm_df.replace(to_replace=cdgm_catalog, value='CDGM', inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Table 6. CDGM Opto-thermal Coefficients Sorted by Refractive Index'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(cdgm_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))
    cdgm_df.sort_values(by='gls', inplace=True)
    display(Latex('\\clearpage\\begin{center}Table 7. CDGM Opto-thermal Coefficients Sorted by Glass Name'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(cdgm_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))
    display(Latex('\\clearpage'))
else:
    display(cdgm_df)
```

```python
# Now merge all materials and print table sorted by refractive index
allgls_df = ohara_df.merge(schott_df, how='outer').merge(schott_sil_df, how='outer'). \
                     merge(cdgm_df, how='outer').sort_values(by=['nd', 'vd'], ascending=[1, 0])
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Table 8. Schott, Ohara and CDGM Opto-thermal Coefficients Sorted by $n_d$'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C)\end{{center}}'))
    display(Latex(allgls_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$dn/dT [1/K]\\times10^{-6}$', 
                                           '$\\gamma$ $[1/K]\\times10^{-6}$'],
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dndT': dndT_formatter, 
                                               'opto_therm_coeff': opto_therm_coeff_formatter,})))
    display(Latex('\\begin{center}\\guillemotleft\\guillemotright'
                  '\end{center}\\clearpage'))
```

<!-- #region -->
# References

<dt>1. Technical Information <a href='https://www.schott.com/d/advanced_optics/3794eded-edd2-461d-aec5-0a1d2dc9c523/1.1/schott_tie-19_temperature_coefficient_of_refractive_index_eng.pdf/1.5/schott_tie-32_thermal_loads_on_optical_glass_us.pdf' name="SchottTIE19">(Schott TIE19)</a></dt>
<dd>Schott Advanced Optics</dd>
<dd><i>TIE-19 Temperature Coefficient of the Refractive Index</i></dd>
<dd>Schott Inc., <b>2016</b></dd>


<dt>2. Proceedings of SPIE <a href='https://doi.org/10.1117/12.2036112' name="Reshidko2013">(Reshidko 2013)</a></dt>
<dd>Reshidko, D. and Sasián, J.</dd>
<dd><i>Method of calculation and tables of optothermal coefficients and thermal diffusivities for glass</i></dd>
<dd>Sasián, J. &amp; Youngworth, R. N. <i>(ed.)</i></dd>
<dd>Optical System Alignment, Tolerancing, and Verification VII</dd>
<dd>SPIE, <b>2013</b>, Vol. 8844, pp. 52 - 62</dd>


<!-- #endregion -->
