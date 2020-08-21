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
Optical designers often refer to charts or tables of available optical glasses in order to select glasses based on various properties or criteria. These properties/criteria may include refractive index and dispersion or relative partial dispersion, other optical or mechanical properties, opto-thermal coefficients and environmental resistance.

This notebook shows how to use the ZemaxGlass toolbox to compile and format tables of various glass properties, specifically the relative partial dispersion deviation $\Delta P_{g,F}$ \cite{SchottTIE29}. This property is usually provided in the Zemax glass catalogues, but the origin of the data is not clear. It is possible that it has been calculated using a standard method, or that different normal lines are perhaps used by different glass manufactureres.

Glasses from [Schott](https://www.schott.com/english/index.html), [Ohara](https://www.ohara-gmbh.com/en/ohara.html) and [CDGM](http://www.cdgm.eu/CDGM/CDGM.html) are included in the example tables listed below.



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
```

```python
# Read the current catalogs filtering for environmetally safe glasses that are standard or preferred status
# Read the Ohara catalog
ohara_catalog = 'ohara_20200623'
ohara = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, 
                                     catalog=ohara_catalog, glass_match='S-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
print(ohara.library.keys())

# Read the Schott catalog
schott_catalog = 'schott_20180601'
schott = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, 
                                     catalog=schott_catalog, glass_match='N-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
print(schott.library.keys())

# Read Schott LITHOSIL-Q
schott_sil = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, 
                                     catalog=schott_catalog, glass_match='LITHOSIL',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10)

# Read the CDGM catalogue. Only standard and preferred status materials
cdgm_catalog = 'cdgm_201904'
cdgm = ZemaxGlass.ZemaxGlassLibrary(ZemaxGlass.agfdir, glass_match='H-',
                                     catalog=cdgm_catalog,
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
print(cdgm.library.keys())

```

```python
# Extract refractive index, abbe dispersion and relative partial dispersion from the catalogs, sort by 
# relative partial dispersion dPgf
ohara_df = ohara.asDataFrame(fields=['nd', 'vd', 'dpgf']).sort_values(['dpgf', 'nd'], ascending=[1, 1])
# Replace catalog name with just Ohara
ohara_df.replace(to_replace=ohara_catalog, value='Ohara', inplace=True)

# Extract Schott data
schott_df = schott.asDataFrame(fields=['nd', 'vd', 'dpgf']).sort_values(['dpgf', 'nd'], ascending=[1, 1])
# Replace catalog name with just Schott
schott_df.replace(to_replace=schott_catalog, value='Schott', inplace=True)

# Schott LITHOSIL-Q
# This material is obsolete, but it has catalogue values for the behaviour of fused silica with temperature
schott_sil_df = schott_sil.asDataFrame(fields=['nd', 'vd', 'dpgf']).sort_values(['dpgf', 'nd'], ascending=[1, 1])
schott_sil_df.replace(to_replace=schott_catalog, value='Schott', inplace=True)

# Extract CDGM data
cdgm_df = cdgm.asDataFrame(fields=['nd', 'vd', 'dpgf']).sort_values(['dpgf', 'nd'], ascending=[1, 1])
# Replace catalog name with just CDGM
cdgm_df.replace(to_replace=cdgm_catalog, value='CDGM', inplace=True)

# Now merge all materials into a single dataframe
allgls_df = ohara_df.merge(schott_df, how='outer').merge(schott_sil_df, how='outer'). \
                     merge(cdgm_df, how='outer').sort_values(by=['dpgf', 'nd'], ascending=[1, 1])
```

```python
def n_d_formatter(n_d):
    return '%10.4f' % n_d
def nu_d_formatter(nu_d):
    return '%10.2f' % nu_d
def dpgf_formatter(dpgf):
    return '%6.5f' % dpgf

if latex_flag:
    display(Latex('\\clearpage\\begin{center}Table 1. Schott, Ohara and CDGM Dispersion Sorted by $\Delta P_{g,F}$'
                 '\end{center}'))
    display(Latex(allgls_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Manufacturer', 'Glass','$n_d$', '$\\nu_d$', 
                                           '$\Delta P_{g,F}$'], 
                                   formatters={'nd': n_d_formatter, 'vd': nu_d_formatter,
                                               'dpgf': dpgf_formatter}))) 
                                               
    
else:
    display(allgls_df)
```

# References

(<a id="cit-SchottTIE29" href="#call-SchottTIE29">Advanced, 2016</a>) {Schott Advanced, ``TIE-29 Refractive Index and Dispersion'', Schott Inc., number: ,   2016.  [online](https://www.schott.com/d/advanced_optics/02ffdb0d-00a6-408f-84a5-19de56652849/1.2/tie_29_refractive_index_and_dispersion_eng.pdf)



<!-- #raw -->
% The raw latex commands in this cell will generate a second reference section, which will work if this
% notebook is downloaded as .tex and compiled as usual (pdflatex, bibtex, pdflatex, pdflatex).
% Some editing of the .tex file will be necessary to remove the first reference section or deal with
% other possible minor issues.
\bibliographystyle{unsrt}
\bibliography{biblio}
<!-- #endraw -->

```python

```
