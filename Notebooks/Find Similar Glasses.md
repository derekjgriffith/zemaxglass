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

# Motivation
The optical design process is a search for glasses, element geometries and placements that, taken together, perform a specific optical task. Finding alternative glass combinations is a very important part of the process. One alternative, if rather crude, is to perform manual substitutions of glasses in an existing design with similar glasses from the same or an alternative glass manufacturers catalogue.

This notebook shows a search for similar glasses from the catalogues from the catalogues of multiple manufacturers, including Schott, CDGM, Ohara and Sumita.

```python
# General imports
import numpy as np
import ZemaxGlass as zg
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
latex_flag = False  # Set this True when downloading .tex for reporting purposes
# Choose wavelength and temperature range (for computation of the opto-thermal coefficients)
wv_lo = 450.0  # nm
wv_hi = 850.0  # nm
temp_lo = -10.0  # deg C
temp_hi = +40.0  # deg C
```

```python
# Read the current catalogs filtering for environmetally safe glasses that are standard or preferred status
# Read the Ohara catalog
ohara_catalog = 'ohara_20200623'
ohara = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=ohara_catalog, glass_match='S-', glass_exclude='.*[A-Z]$',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])

# Read the Schott catalog
schott_catalog = 'schott_20180601'
schott = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=schott_catalog, glass_match='N-', glass_exclude='.*HT|.*ultra',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])

# Will also want to include fused silica, which can be obtained from the Schott catalog
schott_litho = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=schott_catalog, glass_match='LITHOSIL',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10)


# Read the CDGM catalogue. Only standard and preferred status materials
cdgm_catalog = 'cdgm_201904'
cdgm = zg.ZemaxGlassLibrary(zg.agfdir, glass_match='H-', glass_exclude='.*\*$|.*[ABCDT]$',
                                     catalog=cdgm_catalog,
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])

sumita_catalog = 'sumita_20200803'
sumita = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=sumita_catalog, glass_match='K-', glass_exclude='.*(M)',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])

nikon_catalog = 'nikon-hikari_201911'
nikon = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=nikon_catalog, glass_match='J-', # glass_exclude='.*(M)',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])

# Merge the catalogues into a single library
gls_lib = ohara.merge(schott, inplace=False).merge(schott_litho, inplace=False).merge(cdgm, 
                              inplace=False).merge(sumita, inplace=False).merge(nikon, inplace=False)
gls_lib.abbreviate_cat_names()
gls_lib.add_opto_thermal_coeff(temp_lo, temp_hi)
print(gls_lib)
```

```python
# Find glasses near Schott N-BK7 at the d-line refractive index
nearest_gls_a = gls_lib.find_nearest_gls(catalog='Sc', glass='N-BK7', criteria=['nd'], percent=True)
display(nearest_gls_a.head(10))  # Print 10 nearest
```

```python
# Find glasses near to Schott N-BK7 based on the RMS difference of refractive index over the whole wavelength region
# Note that the ordering is slightly different for the two cases
print(f'Wavelength region is {wv_lo} nm to {wv_hi} nm')
nearest_gls_b = gls_lib.find_nearest_gls(catalog='Sc', glass='N-BK7', criteria=['n_rel'], percent=True)
display(nearest_gls_b.head(10))  # Print 10 nearest
```

```python
# Add the n_d and opto-thermal coefficient to one of these tables
gls_lib.supplement_df(nearest_gls_a, ['nd', 'opto_therm_coeff'], inplace=True)
display(nearest_gls_a.head(20))
```

```python

```
