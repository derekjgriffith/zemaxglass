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

```python
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 35)
import ZemaxGlass as zg
%load_ext autoreload
%autoreload 1
%aimport ZemaxGlass
```

```python
efl = 400.0  # mm, focal length of the optical design being considered
focal_ratio = 4.8 
wv = np.array([460.0, 650.0, 825.0, 1100.0, 1650.0])  # nm, the wavelengths of interest, including e and C lines
wv_lo = wv.min()
wv_hi = wv.max()
temp_lo = -10.0  # deg C
temp_hi = +30.0  # deg C
delta_temp = temp_hi - temp_lo
i_wv_0 = 2  # The third wavelength is the reference/primary wavelength
n_wv = wv.size
wv_0 = wv[i_wv_0]
```

```python
# Read in the Ohara catalog and filter for preferred and standard glasses starting with S-
ohara_catalog = 'ohara_20200623'  # filter with S-
schott_catalog = 'schott_20180601'  # filter with N-
hikari_catalog = 'nikon-hikari_201911'  # no filter
# Pick one of the catalogs and read the data with relevant selection filters
pick_catalog = ohara_catalog
gls_lib = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=pick_catalog, glass_match='S-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=[0, 1])
gls_lib.abbreviate_cat_names()
# Calculate opto-thermal coefficients and add to the glass library
gls_lib.add_opto_thermal_coeff(temp_lo, temp_hi, wv_ref=wv_0)
# Filter out materials with extreme opto-thermal coefficients
```

```python
# Establish the glass combination problem
gc = zg.GlassCombo(wv, i_wv_0, [gls_lib], [3], [1.0], efl, temp_lo=-10.0, temp_hi=40.0)
```

```python
# Calculate a suitable Buchdahl alpha for this problem
gc.buchdahl_find_alpha(show_progress=True)
```

```python
print(f'Best fit mean Buchdahl alpha for this problem is {gc.buchdahl_alpha:7.5f}')
```

```python
# Calculate the Buchdahl dispersive power function coefficents eta
gc.buchdahl_fit_eta()
print(gc.eta_per_grp)
```

```python
# Set up the dask.distributed computing client
daclient = gc.dask_client_setup()
```

```python
display(daclient)
```

```python
%time result = gc.dask_run_de_albuquerque(daclient)
```

```python
# Eliminate combinations with large amounts of (absolute) optical power
result = result[result['f_1'] < 9.0]
# Select for small color aberration
result = result[result['f_2'] < 0.01]
# Select for small opto-thermal sensitivity
#result = result[np.abs(result['f_4']) < 0.2]
```

```python
# Sorting by f_2 is by color correction potential
result.sort_values(by='f_1').head(35)
```

```python
gc.dask_client_shutdown(daclient)
```

```python
gc.sum_abs_pow_limit = 8.0
```

```python

```
