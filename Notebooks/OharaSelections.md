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

# Ohara Glass Selections
The purpose of this notebook is to create a number of Zemax glass catalog files (.agf files) with subsets of the current Ohara glass catalog. The subsets are based on a number of selection criteria. Important considerations include the following:
- The current status of the glass in terms of production and possible obsolescence that may mandate a redisgn of the optical system.
- The generalised relative partial dispersion and generalised Abbe number (dispersion) of the glass. For shared aperture VIS/NIR/SWIR systems, the generalised dispersion and relative partial dispersion considered over multiple bands becomes relevant. When a glass is paired with another glass type in a cemented doublet, the color correction is dependent on these parameters. In general, a minimal difference in generalised relative partial dispersion, coupled with a maximal difference in 
- The physical properties of the glass, especially the CTE when the glass is paired with another glass in a cemented doublet or triplet.
- The cost of the glass, although in the present context, this is less of an issue than the above considerations.

This selection process starts with the full Zemax Ohara catalog in the Glasscat folder. The following steps are applied to arrive at the best potential glass combinations for VIS/SWIR doublets.

1. The full Ohara catalog is loaded.
2. Glasses with names starting with S- are selected and all others thus removed. 
3. Generalised dispersion and relative partial dispersion values are calculated for every glass in both the VIS and in the SWIR regions.
4. Color-correction potential is computed for every pair of glass types in the VIS and also in the SWIR.
5. Glass pairs are ordered by the product of the VIS and SWIR color-correction potential.
6. Glass pairs are eliminated if the difference in CTE exceeds an acceptable threshold.
7. Glass pairs with color-correction potential below a minimum threshold are eliminated.
8. For each glass remaining in the list, a catalog of high potential possible pair glasses is compiled and written to a file, where the name of the file is the glass in question e.g. S-BSL7.agf. When trying to find a good pair glass for a specific glass, the adjacent glass is confined for substitution to the pair catalog for that specific glass. This means that the first glass in the pair does not undergo optimization substitution.  


```python
import numpy as np
import ZemaxGlass
import re
import pandas as pd
%load_ext autoreload
%autoreload 1
%aimport ZemaxGlass
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_rows', 100)
```

```python
# Read only the Ohara catalog, filtering for only those glasses that start with 'S-'
ohara = ZemaxGlass.ZemaxGlassLibrary(r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', catalog='ohara', glass_match='S-',
                                     wavemin=430.0, wavemax=1700.0, degree=10)
```

```python
# Print the names of the resulting glasses
ohara.library['ohara'].keys()
```

```python
# Get pairwise ranking for color correction
gpair = ohara.get_pair_rank_color_correction(wv_centre=[0.55, 1.3], wv_x=[0.436, 0.780], wv_y=[0.486, 0.850], 
                                         wv_lo=[0.486, 0.850], wv_hi=[0.656, 1.7], as_df=True)
```

```python
# Display the pair table
gpair
```

```python
# Add the thermal coefficient of expansion and the relative cost
f = ohara.supplement_df(gpair, ['tce', 'nd'])
```

```python
# Select the combinations with tce difference less than a threshold
g = f[np.abs(f['tce1']-f['tce2']) < 1.5]
```

```python
# Filter out high merit combos
h = g[g['merit']>1e6]
```

```python
# Prefer combos with substantial index step
m = h[np.abs(h['nd1']-h['nd2'])>.15]
```

```python
m[m['gls1']=='S-NBH56']
```

```python
m
```

```python

```
