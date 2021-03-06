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

# Testing of the ZemaxGlass Module
ZemaxGlass is Python 3 module for reading, manipulating, analysing and writing of Zemax glass catalog (.agf) text files. This notebook shows usage examples.

Note that the notebooks included with this module are paired with .md files using Jupytext. Only the .md files are under revision control. Install Jupytext to load and run these notebooks.

```python
import numpy as np
import ZemaxGlass
%load_ext autoreload
%autoreload 1
%aimport ZemaxGlass
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
all_glasses = ZemaxGlass.ZemaxGlassLibrary(r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', wavemin=430.0, 
                                       wavemax=1700.0, degree=10, discard_off_band=True)
```

```python
print(all_glasses.cat_encoding)
```

```python
wv=np.linspace(0.9, 1.7, 20)
wv_center = 1.3
alpha = 0.4
print(wv)
cat, glass, indices = all_glasses.get_indices(wv, catalog='ohara', glass='S-BSL7')
cat, glass, n_center = all_glasses.get_indices(wv_center, catalog='ohara', glass='S-BSL7')
print('-', cat, glass, '-')
print(indices)
print(n_center)
```

```python
fit = ZemaxGlass.buchdahl_fit(wv, indices, wv_center, n_center, alpha)
print(fit)
fit_alpha = ZemaxGlass.buchdahl_fit_alpha(wv, indices, wv_center, n_center)
print(fit_alpha)
buch_omega = ZemaxGlass.buchdahl_omega(wv, wv_center, alpha)
buch_indices = ZemaxGlass.buchdahl_model(wv, wv_center, n_center, alpha, fit[0], fit[1], fit[2])
print(buch_omega)
print(buch_indices)
plt.plot(buch_omega, buch_indices)
```

```python
# Try a different apha
alpha = 0.0612
print('Alpha = ', alpha)
fit = ZemaxGlass.buchdahl_fit(wv, indices, wv_center, n_center, alpha)
print(fit)
buch_omega = ZemaxGlass.buchdahl_omega(wv, wv_center, alpha)
buch_indices = ZemaxGlass.buchdahl_model3(wv, wv_center, n_center, alpha, fit[0], fit[1], fit[2])
print(buch_omega)
print(buch_indices)
```

```python
plt.figure(figsize=(10,10))
plt.plot(buch_omega, buch_indices)

```

```python
new_fit = ZemaxGlass.buchdahl_find_alpha(wv, indices, wv_center, n_center, gtol=1.0e-7)
print('Finishing fit', new_fit['x'])

```

```python
fitx = new_fit['x']
buch_omega_new = ZemaxGlass.buchdahl_omega(wv, wv_center, fitx[0])
buch_indices_new = ZemaxGlass.buchdahl_model(wv, wv_center, n_center, fitx[0], fitx[1], fitx[2], fitx[3])
```

```python
plt.figure(figsize=(10,10))
plt.plot(buch_omega_new, buch_indices_new, buch_omega, buch_indices)
```

```python
ohara = ZemaxGlass.ZemaxGlassLibrary(r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', catalog='ohara', wavemin=430.0, 
                                       wavemax=1700.0, degree=10)
cat, glass, buch_fits = ohara.buchdahl_find_alpha(wv, wv_center, show_progress=True)
```

```python
# Get the mean alpha parameter for the unabridged Ohara catalogue
buch_fits[:,0].mean()
```

```python
heraeus = ZemaxGlass.ZemaxGlassLibrary(r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', catalog='heraeus', wavemin=430.0, 
                                       wavemax=1700.0, degree=10, discard_off_band=True)
her_cat, her_glass, her_buch_fits = heraeus.buchdahl_find_alpha(wv, wv_center, show_progress=True)
```

```python
buch_fits[:,0].mean()
```

```python
print(f'{heraeus.cat_comment["heraeus"]}')
print(f'{heraeus.cat_encoding["heraeus"]}')
print(heraeus.library.keys())
```

```python
# Print the recorded .agf text for a Heraeus material
print(heraeus.library['heraeus']['HOMOSIL101_HERASIL102']['text'])
```


```python
all_glasses.cat_comment
```

```python
all_glasses.library['schott']['N-F2']
```

```python
all_glasses.plot_dispersion_ratio('S-BSL7', 'ohara', 'N-BK7', 'schott', subtract_mean=True)
```

```python
all_glasses.plot_dispersion('S-BSL7', 'ohara')
```

```python
all_glasses.plot_dispersion('N-BK7', 'schott', polyfit=True, fiterror=True)
```

```python
ohara = ZemaxGlass.ZemaxGlassLibrary(r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', catalog='ohara', wavemin=430.0, 
                                       wavemax=1700.0, degree=10)
```

```python
ohara.cat_comment
```

```python
ohara.plot_catalog_property_diagram('all')  #, prop1='n0', prop2='n1')
```

```python
all_glasses.plot_catalog_property_diagram('all', prop1='nd', prop2='dispform')
```

```python
plt.plot(all_glasses.library['schott']['F2']['it']['wavelength_np'], 
         all_glasses.library['schott']['F2']['it']['transmission_np'])
```

```python
all_glasses.plot_temperature_dependence('N-BK7', 'schott', 900.0, [-40, 65])
```

```python
ohara = ZemaxGlass.ZemaxGlassLibrary(dir=r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', catalog='ohara', discard_off_band=True)
```

```python
len(ohara.library['ohara'].keys())
```

```python
ohara.pprint()
```

```python
all_glasses.plot_dispersion('F_SILICA', 'misc', polyfit=True, fiterror=True)
```

```python
all_glasses.library['misc'].keys()
```

```python
all_glasses.plot_dispersion_ratio('N-SK5', 'schott', 'N-PK51', 'schott')
```

```python
all_glasses.plot_dispersion_ratio('F2', 'schott', 'N-F2', 'schott')
```

```python
ohara.pprint()
```

```python
glasses, nd_ohara = all_glasses.get_indices(catalog='ohara')
glasses, abbe_ohara = all_glasses.get_abbe_number(catalog='ohara')
glasses, rel_pd_ohara = all_glasses.get_relative_partial_dispersion(catalog='ohara')
```

```python
plt.figure(figsize=(16,10))
plt.plot(-abbe_ohara, nd_ohara, 'o')
plt.grid()
```

```python
plt.figure(figsize=(16,10))
plt.plot(-abbe_ohara, rel_pd_ohara, 'o')
plt.grid()
```

```python
glasses, nd_schott = all_glasses.get_indices(catalog='schott')
glasses, abbe_schott = all_glasses.get_abbe_number(catalog='schott')
glasses, rel_pd_schott = all_glasses.get_relative_partial_dispersion(catalog='schott')
```

```python
plt.figure(figsize=(16,10))
plt.plot(-abbe_schott, rel_pd_schott, 'or')
plt.plot(-abbe_ohara, rel_pd_ohara, 'xb')
plt.legend(['Schott', 'Ohara'])
plt.grid()
```

```python
# Calculate the color correction potential for all glass pairs in the Ohara catalogue
glass1, glass2, color_correction_rank = all_glasses.get_pair_rank_color_correction(catalog='ohara')
```

```python
glass_combo_list[-10:]
```

```python
glass_combo_list[0:10]
```

```python
np.char.add(np.char.add(glass1, ' + '), glass2)[:10]
```

```python
color_correction_rank
```

```python
all_glasses = ZemaxGlass.ZemaxGlassLibrary(r'C:/Users/EZGRIF/Documents/Zemax/Glasscat/', wavemin=430.0, 
                                       wavemax=1700.0, degree=10, discard_off_band=True)
```

```python
glass_combo_list = [glass1 + ' ' + glass2 for glass1 in glass_list for glass2 in glass_list]
```

```python
glass_combo_list
```

```python
a = np.random.normal(size=(20,20))
```

```python
rankorder = a.flatten().argsort()
```

```python
a.flatten()[rankorder]
```

```python
np.vstack(([1,2,3],[4,5,6]))
```

```python

```
