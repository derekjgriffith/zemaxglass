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

# Sumita Moulding and Polishing Glasses
Sumita manufactures glasses that can be placed, roughly speaking, into three classes. There is a set of Sumita glasses that are for precision moulding of (typically relatively small) optical components. Then there is a set of glasses that are specified to be used with conventional lens grinding and polishing processes. The transformation temperature of the glasses for moulding is typically lower than that for the other glasses, but the forms of delivery for the two sets of glasses are likely to be different.

The current (2020-08) environmentally friendly (lead and arsenic-free) glass offerings from Sumita are prefixed with `K-`.

The glasses for conventional processing can be split into two sets as well, namely the "Schott equivalents" and the glasses that are not intended to be Schott equivalents.

<!-- #region -->
## Sumita Glasses for Conventional Processing
The glasses for conventional polishing (Schott equivalent and non-Schott equivalent) are:

K-FK5
K-BK7
K-BPG2
K-PSKn2
K-SK4
K-SK5
K-SK7
K-SK14
K-SK15
K-SK16
K-SK16RH
K-SK18
K-SK18RH
K-SSK1
K-SSK3
K-SSK4
K-SSK9
K-BaF8
K-BaF9
K-BaFn1
K-BaFn3
K-BaSF4
K-BaSF5
K-BaSF12
K-LaK6
K-LaK7
K-LaK8
K-LaK9
K-LaK10
K-LaK11
K-LaK12
K-LaK13
K-LaK14
K-LaK18
K-LaKn2
K-LaKn7
K-LaKn12
K-LaKn14
K-LaF2
K-LaF3
K-LaFn1
K-LaFn2
K-LaFn3
K-LaFn5
K-LaFn9
K-LaFn11
K-LaSKn1
K-LaSFn1
K-LaSFn2
K-LaSFn3
K-LaSFn4
K-LaSFn6
K-LaSFn7
K-LaSFn8
K-LaSFn8W
K-LaSFn9
K-LaSFn10
K-LaSFn14
K-LaSFn16
K-LaSFn17
K-LaSFn21
K-LaSFn22
K-LaSFn23
K-SFLD1
K-SFLD2
K-SFLD4
K-SFLD5
K-SFLD6
K-SFLD8
K-SFLD8W
K-SFLD10
K-SFLD11
K-SFLD14
K-SFLDn3
K-SFLDn3W
K-BOC30
K-GIR79
K-GIR140

The following glasses for conventional processing are not Schott-equivalent:

K-BPG2
K-SFLD1
K-SFLD2
K-SFLD4
K-SFLD5
K-SFLD6
K-SFLD8
K-SFLD8W
K-SFLD10
K-SFLD11
K-SFLD14
K-SFLDn3
K-SFLDn3W
K-BOC30
K-GIR79
K-GIR140

The following conventional process glasses currently (2020-08) have limited or no therm-optic data (these glasses cannot be used if system athermalisation or thermo-optic analysis is required):

K-SK16
K-BaSF12
K-LaK13
K-LaF3
K-LaFn1
K-LaFn2
K-LaFn9
K-SFLD2
K-SFLD5
K-SFLD10


## Sumita Glasses for Moulding

The remaining Sumita glasses are for moulding:

K-CaFK95
K-PFK80
K-PFK85
K-PFK90
K-GFK68
K-GFK70
K-PBK40
K-PBK50
K-PBK60
K-PMK30
K-PSK100
K-PSK200
K-PSK300
K-PSK400
K-PSK500
K-CSK120
K-SKLD100
K-SKLD120
K-SKLD200
K-LaFK50
K-LaFK50T
K-LaFK55
K-LaFK58
K-LaFK60
K-LaFK63
K-LaFK65
K-VC78
K-VC79
K-VC80
K-VC82
K-VC89
K-VC90
K-VC91
K-VC99
K-VC100
K-VC179
K-VC181
K-VC185
K-CD45
K-CD120
K-CD300
K-LCV93
K-LCV161
K-ZnSF8
K-PSFn1
K-PSFn2
K-PSFn3
K-PSFn166
K-PSFn185
K-PSFn190
K-PSFn202
K-PSFn214P
K-PG325
K-PG375
K-FIR98UV
K-FIR100UV

All the current (2020-08) Sumita glasses for moulding appear to have therm-optic data.

The glasses K-FIR98UV and K-FIR100UV are remarkable for their deep UV transmission.
<!-- #endregion -->

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
# Specify Sumita polishing glasses
sumita_catalog = 'sumita_20200803'  # Will read Sumita catalog from 2020-08
sumita_pol_match =  'K-FK5|K-BK7|K-BPG2|K-PSKn2|K-SK4|K-SK5|K-SK7|K-SK14|K-SK15|K-SK16|K-SK16RH|' \
                    'K-SK18|K-SK18RH|K-SSK1|K-SSK3|K-SSK4|K-SSK9|K-BaF8|K-BaF9|K-BaFn1|K-BaFn3|' \
                    'K-BaSF4|K-BaSF5|K-BaSF12|K-LaK6|K-LaK7|K-LaK8|K-LaK9|K-LaK10|K-LaK11|K-LaK12|' \
                    'K-LaK13|K-LaK14|K-LaK18|K-LaKn2|K-LaKn7|K-LaKn12|K-LaKn14|K-LaF2|K-LaF3|K-LaFn1|' \
                    'K-LaFn2|K-LaFn3|K-LaFn5|K-LaFn9|K-LaFn11|K-LaSKn1|K-LaSFn1|K-LaSFn2|K-LaSFn3|' \
                    'K-LaSFn4|K-LaSFn6|K-LaSFn7|K-LaSFn8|K-LaSFn8W|K-LaSFn9|K-LaSFn10|K-LaSFn14|' \
                    'K-LaSFn16|K-LaSFn17|K-LaSFn21|K-LaSFn22|K-LaSFn23|K-SFLD1|K-SFLD2|K-SFLD4|' \
                    'K-SFLD5|K-SFLD6|K-SFLD8|K-SFLD8W|K-SFLD10|K-SFLD11|K-SFLD14|K-SFLDn3|' \
                    'K-SFLDn3W|K-BOC30|K-GIR79|K-GIR140'
```

```python
# Specify Sumita moulding glasses
sumita_mol_match =  'K-CaFK95|K-PFK80|K-PFK85|K-PFK90|K-GFK68|K-GFK70|K-PBK40|K-PBK50|K-PBK60|' \
                    'K-PMK30|K-PSK100|K-PSK200|K-PSK300|K-PSK400|K-PSK500|K-CSK120|K-SKLD100|' \
                    'K-SKLD120|K-SKLD200|K-LaFK50|K-LaFK50T|K-LaFK55|K-LaFK58|K-LaFK60|K-LaFK63|' \
                    'K-LaFK65|K-VC78|K-VC79|K-VC80|K-VC82|K-VC89|K-VC90|K-VC91|K-VC99|K-VC100|' \
                    'K-VC179|K-VC181|K-VC185|K-CD45|K-CD120|K-CD300|K-LCV93|K-LCV161|K-ZnSF8|' \
                    'K-PSFn1|K-PSFn2|K-PSFn3|K-PSFn166|K-PSFn185|K-PSFn190|K-PSFn202|K-PSFn214P|' \
                    'K-PG325|K-PG375|K-FIR98UV|K-FIR100UV' 
```

```python
sumita_polish_gls = zg.ZemaxGlassLibrary(catalog=sumita_catalog, glass_match=sumita_pol_match)
print(sumita_polish_gls)
# Save the polishing glass catalog
sumita_polish_gls.write_agf(filename=os.path.join(zg.agfdir, 'SUMITA_20200803_pol.agf'),
                            cat_comment='Sumita glasses for conventional polishing')
```

```python
sumita_mould_gls = zg.ZemaxGlassLibrary(catalog=sumita_catalog, glass_match=sumita_mol_match) 
print(sumita_mould_gls)
# Save the molding glass catalog
sumita_mould_gls.write_agf(filename=os.path.join(zg.agfdir, 'SUMITA_20200803_mol.agf'),
                          cat_comment='Sumita glasses for molding')
```
