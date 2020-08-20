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
The optical design requirement is for a lighweight lens assembly of overall focal length 500 mm at a focal ratio of f/6. The design is to cover a focal plane of diameter 66 mm having a nyquist frequency of 133 cy/mm. The design should be close to diffraction limited with modulation of 25% or more at the nyquist frequency. The spectral range is 450 nm to 850 nm.

The operating temperature range is -10$^\circ$C to +40$^\circ$C.

Preliminary work has established a 5-element telephoto lens solution with 2 aspheric surfaces. While the manufacturing and assembly tolerances are very tight, the solution meets the optical requirement, except for a large focus shift with temperature. This is largely due to the large, negative opto-thermal coefficient of the leading optical material in the design, which is a low index and low dispersion fluoro-crown glass (e.g. Ohara S-FPL53, Ohara S-FPL55 or Schott N-FK58). Optical glasses in this category all have large, negative opto-thermal coefficients, which are very difficult to balance out using the subsequent materials in the design.

This notebook is an attempt to find alternative glass combinations with a smaller overall opto-thermal defocus, preferably sufficiently small to be within the system depth of focus. However, any reduction of the opto-thermal defocus is considered beneficial as it broadens the range of options available for dealing with the residual defocus.

<!-- #region -->
# Paraxial Layout
The elementary layout of the dioptric solution comprises a classic telephoto with a major front lens group having overall positive optical power and a rear lens group of overall negative power. The telephoto ratio is the ratio between the physical length and the effective focal length. The targeted telephoto ratio is 0.8, which implies a physical length of 400 mm, but telephoto ratio of up to 0.88 (440 mm physical length) will be considered if required.

The paraxial layout in this case comprises a pair of planes representing "thin" lens groups with assigned optical power and separation. Optical power is typically expressed in units of dioptres (symbol $D$, units $m^{-1}$) which is the reciprocal of the focal length expressed in meters. The glass selection process described below uses a specific paraxial layout as an input. 

The two major groups in the paraxial design will each comprise multiple physical lens elements.

The paraxial layout of a telephoto lens of 500 mm EFL at f/6 with optical length of 400 mm is shown below. The optical power of the rear group is the same as that of the front group, but with opposite sign.



![](Figures/ParaxialLayout80percentTelephotoRatioMarkup.pdf)
<!-- #endregion -->

```python
from IPython.core.display import Math, Markdown, Latex, SVG
```

```python
latex = False
if not latex:
    display(SVG(filename='Figures/ParaxialLayout80percentTelephotoRatioMarkup.svg'))
```

<!-- #region -->
# Methods of Glass selection
The glass selection problem is tackled with a variation of the method due to de Albuquerque et al. \cite{Albuquerque2012}, with additional elements taken from Wang et al. \cite{Wang2019} and also from Lim and Park \cite{Lim2016}. The main variation of the de Albuquerque method is that of apportioning optical power to multiple lens groups. During the exposition, a vector $\bar{S}$ is presented which corresponds to the relative weighting of a lens group to the total system power. If the system comprises a $k$ coincident thin lenses, the total power, being the reciprocal of the effective focal length is
\begin{equation}\label{eq:total_power}
\Phi=\sum_{j=1}^k \phi_j=\frac{1}{F}.
\end{equation}

If the overall system power is normalised to 1, this is written as
\begin{equation}\label{eq:total_power_norm}
\hat{\Phi} = F \cdot \Phi = \sum_{j=1}^k \hat{\phi}_j=1.
\end{equation}

If, instead, the first order layout of the optical train comprises multiple, despaced thin lenses (or thin lens groups), then the balance of power is altered according to the (paraxial) marginal ray height at each of the thin lenses as
\begin{equation}\label{eq:total_power_distr}
\Phi = \sum_{j=1}^k \frac{h_j}{h_1} \phi_j,
\end{equation}

where $h_i$ is the marginal ray height at thin lens having power $\phi_i$. This weighting scheme is applicable to axial colour. The treatment for lateral colour could use different weighting factors. The chief 

Likewise if this is normalised to total power of 1, then
\begin{equation}\label{eq:total_norm_power_distr}
\hat{\Phi} = F \cdot \Phi = \sum_{j=1}^k \frac{h_j}{h_1} \hat{\phi}_j=1.
\end{equation}


This is expressed in vector notation as
\begin{equation}\label{eq:vec_total_power}
\bar{S} \cdot \bar{\Phi} = 1,
\end{equation}
where the elements of the vector $\bar{S}$ are $\dfrac{h_j}{h_1}$ and the elements of $\bar{\Phi}$ are the normalised powers $\hat{\phi}_j$ with $j=1\cdots k$.


The methods developed by de Albuquerque et al. \cite{Albuquerque2014,Albuquerque2012,Albuquerque2016} as well as Wang et al. \cite{Wang2019} make use of the Buchdahl dispersion function where the refractive index of a material is written as
\begin{equation}\label{eq:buchdahl}
N(\lambda) = N_0+\sum_{i=1}^{n-1} \nu_i \omega(\lambda)^i,
\end{equation}
where the the refractive index is $N_0$ at the reference wavelength of $\lambda_0$ and the coefficents $\nu_i$ are applied to a power series in $\omega$ which is related only to wavelength as
\begin{equation}\label{eq:omega}
\omega = \frac{\Delta \lambda}{1+\alpha \Delta \lambda},
\end{equation}
in which $\Delta \lambda = \lambda-\lambda_0$ is the wavelength deviation from the reference wavelength $\lambda_0$. The constant $\alpha$ can be tuned to a particular collection of glasses for best mean fitting of the Buchdahl function to that specific collection over a specific wavelength region. The universal mean for optical glass catalogs (Schott, Ohara etc.) is sometimes taken as $\alpha=2.5$. However, this notebook illustrates the calculation of a tuned $\alpha$ for a selection of Ohara materials.

Equation \ref{eq:buchdahl} can be used for an exact fit of the Buchdahl dispersion function at $n$ wavelengths, denoted $\lambda_i$ with $i=1\cdots n$.

By manipulation of Equation \ref{eq:buchdahl}, the Buchdahl dispersive power function of an optical material can be written as
\begin{equation}\label{eq:dispersive_power}
D(\lambda)=\sum_{i=1}^{n-1} \eta_i \omega(\lambda)^i,
\end{equation}
where, if $\Delta N(\lambda)=N(\lambda)-N_0$, having defined
\begin{equation}\label{eq:}
D(\lambda)=\frac{\Delta N(\lambda)}{N_0 - 1}
\end{equation}
and
\begin{equation}\label{eq:eta_i}
\eta_i = \frac{\nu_i}{N_0-1}.
\end{equation}

The lens element combination is to incorporate $k$ glass materials with index $j=1\cdots k$. The $i^{\mathrm{th}}$ power coefficient of the dispersive power function is thus $\eta_{ij}$ and the matrix of coefficients is denoted (with one column per glass) as

\begin{equation}\label{eq:eta_bar}
\bar\eta=\left[
\begin{array}{cccc}
   \eta_{11}     & \eta_{12}      & \cdots & \eta_{1k} \\
   \eta_{21}     & \eta_{22}      & \cdots & \eta_{2k} \\
   \vdots        & \vdots         & \ddots & \vdots    \\
   \eta_{(n-1)1} &  \eta_{(n-1)2} & \cdots & \eta_{(n-1)k}
\end{array}
\right].
\end{equation}
<!-- #endregion -->

```python
# Major code imports
import numpy as np
import pandas as pd
# Set maximum rows to display for pandas
pd.set_option('display.max_rows', 100)

import matplotlib.pyplot as plt
%matplotlib inline
import ZemaxGlass as zg
import copy
%load_ext autoreload
%autoreload 1
%aimport ZemaxGlass
from IPython.core.display import Math, Markdown, Latex, SVG
daclient = None  # Dask client
```

## Step 1 - Provide Main System Requirements, Wavelengths and Glass Library

As input data for the method the designer must provide the system effective focal length  $F$ (variable `efl`), the f-number $f$ (`focal_ratio`), the $n$ wavelengths (`wv`) that cover the desired spectral range, and the number of the primary wavelength $\lambda_0$ (`i_wv_0`). A glass library (`gls_lib`) must be provided together with the desired number of glasses in the combination $k$ (`k_gls`).

We will provide four wavelengths in the range 460 nm to 800 nm, one of which will be the primary wavelength.

```python
# First a search is conducted for the front group, comprising 3 lens elements
efl = 361.8  # mm, focal length of the optical design being considered, this is the front group
focal_ratio = 4.8 
wv = np.array([460.0, 546.074, 656.2725, 800.0])  # nm, the wavelengths of interest, including e and C lines
wv_lo = wv.min()
wv_hi = wv.max()
temp_lo = -10.0  # deg C
temp_hi = +40.0  # deg C
delta_temp = temp_hi - temp_lo
i_wv_0 = 2  # The third wavelength is the reference/primary wavelength
n_wv = wv.size
wv_0 = wv[i_wv_0]
```

```python
# Read in one or more glass catalogs and merge into a single catalog
ohara_catalog = 'ohara_20200623'  # filter with S-
schott_catalog = 'schott_20180601'  # filter with N-
hikari_catalog = 'nikon-hikari_201911'  # no filter
cdgm_catalog = 'cdgm_201904'  # filter with H-
hoya_catalog = 'hoya'
# Read catalogs with relevant selection criteriua
status = [0, 1]  # 0 = standard, 1 = preferred glass status
ohara = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=ohara_catalog, glass_match='S-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=status)
schott = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=schott_catalog, glass_match='N-', glass_exclude='.*HT|.*B',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=status)

hikari = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=hikari_catalog, glass_match='J-', glass_exclude='.*-U|.*-V',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=status)

cdgm = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=cdgm_catalog, glass_match='H-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=status)

hoya = zg.ZemaxGlassLibrary(zg.agfdir202002, 
                                     catalog=hoya_catalog, glass_exclude='MP-|MC-',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10, select_status=status)

# Will also want to include fused silica, which can be obtained from the Schott catalog
schott_litho = zg.ZemaxGlassLibrary(zg.agfdir, 
                                     catalog=schott_catalog, glass_match='LITHOSIL', #  glass_exclude='.*HT|.*B',
                                     wavemin=wv_lo, wavemax=wv_hi, degree=10)  #, select_status=status)

# Select and merge catalogs if required
# gls_lib = ohara.merge(cdgm, inplace=False).merge(schott, inplace=False)
# gls_lib = ohara.merge(schott, inplace=False).merge(hikari, inplace=False)
# gls_lib = ohara.merge(schott_litho, inplace=False).merge(cdgm, inplace=False).merge(schott, 
#                            inplace=False).merge(hikari, inplace=False).merge(hoya, inplace=False)
gls_lib = ohara.merge(schott_litho, inplace=False).merge(schott_litho, inplace=False).merge(schott, 
                           inplace=False)
# gls_lib = ohara.merge(hoya, inplace=False).merge(schott_litho, inplace=False)

#gls_lib = ohara.merge(schott_litho, inplace=False).merge(schott, inplace=False).merge(hoya, inplace=False)
schott.abbreviate_cat_names()
ohara.abbreviate_cat_names()
# Abbreviate catalog names just to reduce printed width of tables.
gls_lib.abbreviate_cat_names()
print(gls_lib)
# Compute all opto-thermal coefficients
gls_lib.add_opto_thermal_coeff(temp_lo, temp_hi, wv_ref=wv_0)
# Filter out materials with extreme opto-thermal coefficients if required
# No filtering for this problem
```

```python
# Establish the glass combination problem, 
# Give a default buchdahl alpha parameter, which will be overriden by buchdahl_find_alpha(), if run
gc = zg.GlassCombo(wv, i_wv_0, [schott, schott], [3, 2], [1.0, 0.276], efl=500.0, buchdahl_alpha=2.16, 
                   sum_abs_pow_limit=10.0, max_delta_f=0.5,
                   temp_lo=-10.0, temp_hi=40.0, max_result_rows=3000000)
```

```python
# Calculate a suitable Buchdahl alpha for this problem
%time gc.buchdahl_find_alpha(show_progress=True)
```

```python
# Display the mean Buchdahl alpha parameter value
display(Latex(f'The computed/chosen Buchdahl $\\alpha$ parameter for this glass library is {gc.buchdahl_alpha:4.6f}'))
```

```python
# Calculate the Buchdahl dispersive power function coefficients eta
gc.buchdahl_fit_eta()
```

## Process All Glass Combinations
In this step, all possible combinations of $k$ glasses are processed. If there are $N_g$ glasses in the library (a glass library in this case is defined as a set of glasses that may be selected from the catalogs of one or more glass manufacturers), then the total number of $k$-glass combinations is
\begin{equation}\label{eq:total_gls_comb}
{{N_g}\choose{k}} = \frac{N_g !}{k!(N_g - k)!} 
\end{equation}

It is important to note here that the chromatic aberration correction depends only on the $k$ glass combination and the optical power assigned to each glass. It does not depend on the order in which the glasses are placed in the optical design. The order of the glasses and lens bending (change of lens shape while maintaining optical power constant) are variables the optical designer can exploit to correct other aberrations.  

The normalised optical power $\hat{\phi}_j$ of the three lenses making up each $k$-material combination is computed. For example, in the case of $k=3$ the optimal power distribution vector $\hat{\bar \Phi} = (\hat{\phi}_1, \hat{\phi}_2, \hat{\phi}_3)$ to the three lenses is generally computed using a least squares method as
\begin{equation}\label{eq:Phibarhat}
\hat{\bar \Phi} = \left( \bar{G}^t \cdot \bar{G} \right)^{-1} \cdot \bar{G}^t \cdot \hat{e}
\end{equation}

where $\bar{G}$ is defined as
\begin{equation}\label{eq:Gbar}
\bar{G}=
\left[\begin{array}{c}
\bar{S}\\
\Delta\bar{\Omega}\cdot\bar{\eta}
\end{array}\right]
\end{equation}
with $\bar{S}$ a row vector (order $1 \times k$) of the relative marginal ray heights (see Equation \ref{eq:total_norm_power_distr}) and $\hat{e}$ a column vector with 1 at the top and zeros below (order $n\times 1$)

Depending on the number of glasses in the library and the number of glasses in the combination, this step can take quite a long time.

### Opto-thermal Effects
In addition to computation of the de Albuquerque metrics $F_1$ (sum of absolute normalised lens powers) and $F_2$ (modulus of the normalised chromatic power shift), the lens power shift due to a change in temperature is also computed.

The opto-thermal coefficient of a glass material is defined \cite{Reshidko2013} as
$$
\gamma = \frac{\frac{\mathrm{d}N}{\mathrm{d}T}}{N-1}-\alpha_l,
$$
where $\frac{\mathrm{d}N}{\mathrm{d}T}$ is the thermal coefficient of refractive index (units $K^{-1}$), $N$ is the nominal refractive index (at some reference wavelength, temperature and pressure) and $\alpha_l$ is the linear coefficient of thermal expansion (CTE, not the Buchdahl $\alpha$) of the glass (in $m/m/K$ which is just $K^{-1}$).
The opto-thermal change in focal length $\Delta_T F$ of a single lens element of nominal focal length $F$ is related to the opto-thermal coefficient as
\begin{equation}
\frac{\Delta_T F}{F} = -\gamma \cdot \Delta T.
\label{eq:focal_length_gamma}
\end{equation}
The power of a lens component is just the recprocal of the focal length $F$. That is $F=1/\phi$, leading (by differentiation) to
\begin{equation}\label{eq:therm_power_shift}
\Delta_T \phi=\gamma \cdot \phi \cdot \Delta T
\end{equation}

Here, the total power $\Phi$ of 3 coincident thin lenses is taken as the sum of their powers, that is
\begin{equation}\label{eq:sum_of_powers}
\Phi = \sum_{j=1}^k \phi_j.
\end{equation}

The opto-thermal change in total power $\Delta_T\Phi$ is taken as the sum of the opto-thermal changes in power of the components $\Delta_T \phi_j$ so
\begin{equation}\label{eq:opto_therm_power}
\Delta_T \Phi = \Delta T \sum_{j=1}^k \gamma_j \cdot \phi_j
\end{equation}
where $\gamma_j$ is the opto-thermal coefficient for the material of lens element $j$.

Usually, the metric of interest is the focal shift over a particular temperature range. This is computed as
\begin{equation}\label{eq:delta_F_therm}
\Delta_T F = -F^2 \cdot \Delta_T \Phi = -F^2 \cdot \Delta T \sum_{j=1}^k \gamma_j \cdot \phi_j.
\end{equation}

The de Albuquerque method calculates the vector of normalised powers ($\hat{\phi}_j$ where $\sum \hat{\phi}_j=1$), denoted $\hat{\bar{\Phi}}$. Therefore $\hat{\phi}_j=F\phi_j$ and 
\begin{equation}\label{eq:delta_F_therm_rel}
\Delta_T F = -F \cdot \Delta T \sum_{j=1}^k \gamma_j \cdot \hat{\phi}_j.
\end{equation}

In order for the lens combination to be well athermalised, the change in focal length $\Delta_T F$ should be within the depth of focus of the lens, which is approximated as
\begin{equation}\label{eq:DOF}
\varepsilon = \pm 2 \lambda f^2
\end{equation}

where $f$ (`focal_ratio`) is the focal ratio of the lens and $\lambda$ can be taken as the reference wavelength $\lambda_0$. 

### Thermo-Chromatic Figure of Merit
Actually, the total thermo-chromatic focus shift must be considered in relation to the depth of focus. While there is possibly a residual statistical correlation of chromatic and opto-thermal focus shift, the root-mean-square is taken as a figure of merit, that is
\begin{equation}\label{eq:DeltaFtotal}
\Delta F = \sqrt{(\Delta_T F)^2 + (\Delta_C F)^2},
\end{equation}
where $\Delta_C F$ is an absolute measure of the chromatic focal shift. The metric proposed by de Albuquerque makes use of the chromatic change in normalised power $\overline{CCP}$, defined as
\begin{equation}\label{eq:CCP}
\overline{CCP}=\Delta \bar{\Omega} \cdot \bar{\eta} \cdot \hat{\bar{\Phi}},
\end{equation}
all of which are defined above, and the absolute chromatic focal shift is estimated by scaling the $\overline{CCP}$ by the focal length giving
\begin{equation}\label{eq:Delta_CofF}
\Delta_C F = \| \overline{CCP} \| \cdot F.
\end{equation}







# Correction of Upstream Residuals
A cross-catalog search for 5-glass combinations is very time-consuming. However, the bigger problem is that such a broad search yields lens powers that are difficult to distribute into a working optical design. One approach that can be attempted is to find 3-glass (perhaps 4-glass) combinations starting from the front of the lens system and then to make an extract of the best such combinations. This extract of frozen (in the sense that both the glasses and assigned powers $\hat{\bar{\Phi}}$ are fixed) combinations are then passed into a full search for the next 3 or 4 downstream glasses. This approach greatly reduces the number of combinations that must be processed and also helps to guarantee that the upstream glass combination has a reasonable level of correction, making the downstream search more productive. An important aspect here is that the downstream glass selection process must be constrained to provide the best possible compensation for the residual chromatic errors in the upstream combination(s).

The upstream glass combinations are not separated into groups. Instead, upstream glasses are listed individually for each promising combination, together with absolute powers and weighting factors.

The de Albuquerque process is modified to use absolute lens and group powers instead of relative powers. If there are upstream glass combinations in the process, the downstream lens group or groups are selected for their ability to correct the upstream chromatic residual errors.

The glass selection process for the 5-element telephoto now becomes as follows:
\begin{enumerate}
\item Run the group glass selection process to find promising 3-glass combinations for the front group. The group weighting factor is set to 1 and a large variety of glasses from different manufacturers can be considered.
\item The most promising 3-glass combinations (based on a thermo-chromatic figure of merit) are chosen from the results of the first set. These glass combinations become the upstream combinations for the following downstream search.
\item The glass selection process is run again for the downstream (second) group of 2 glasses. 
  \begin{enumerate}
    \item The absolute weight(s) for the downstream group(s) are provided, as well as the absolute group power or effective focal length.
    \item For each promising upstream combination, all glass combinations in the glass library for the downstream group are tested.
    \item The chromatic residual targets for the downstream group(s) is set to the negative of the residual upstream chromatic errors. The process can be extended to include the opto-thermal defocus.
  \end{enumerate} 
\end{enumerate}

The upstream residual correction search as described here requires that the focal lengths or powers of each group as well as the group power comes into play. The power distribution is specified per glass in the upstream combinations on the basis of the power weights. However, for the purposes of the downstream glass search, the only important metric of the upstream combinations are the chromatic or opto-thermal residuals. These are both expressed in terms of the effective power delta for the whole system.

```python
# Set up the dask.distributed computing client if not already existing
daclient = gc.dask_client_setup(daclient)
```

```python
# Display dask.distributed client information
display(daclient)
```

```python
# Run the de Albuquerque metrics
%time result = gc.dask_run_de_albuquerque(daclient)
```

```python
result.sort_values(by=['f_2']).head(100)
```

```python
# Make some extracts from the resulting best combinations
rex = result.copy()
# Eliminate combinations with large amounts of (absolute) optical power
rex = rex[(rex['f_1'] > 0.0) & (rex['f_1']) <= 9.0]
```

```python
# Select for small color aberration
rex = rex[(rex['f_2'] < 0.3) & (rex['f_2'] > 0.0)]
```

```python
# Select for small opto-thermal sensitivity
# Skip this cell if not required
rex = rex[(rex['f_4'] < 0.2) & (rex['f_4'] > -0.1)]
```

```python
np.abs(np.array([-1, -2, 4.0])/4.0)
```

```python
# Sorting by f_2 is by color correction potential
# Sorting by f_4 is by therm-optic sensitivity

latex_flag = False
rex.sort_values(by=['f_5'], inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Glass Triplet Candidates for VNS Chromatic Performance'
                  f'Wavelength {wv_lo} nm to {wv_hi} nm, Sorted by Chromatic Correction $F_2$\end{{center}}'))
    display(Latex(rex.to_latex(index=False, longtable=True, escape=False,
                                   header=['Cat' ,'Gls', 'Pow',
                                           'Cat' ,'Gls', 'Pow',
                                           'Cat' ,'Gls', 'Pow]',
                                           '$F_1$', '$F_2$ [mm]', '$F_4$ [mm]'],
                                   float_format="%.4f")))
else:
    display(rex.head(100))
```

```python
np.abs(rex['p1']) + np.abs(rex['p2']) + np.abs(rex['p3']) + np.abs(rex['p4'])  # 2.7639579878385847
```

```python
rex.to_excel('TripletOharaSchottHoyaB.xlsx')
```

```python
display(rex)

```

```python
# Plot the 3-tuples on a standard glass map
plt.figure(figsize=(12,12))
gc.plot_combo_usingDataFrame(rex.head(6), df_col_names=[('c1','g1'),('c2','g2'),('c3','g3')])
plt.xlim([100, 15])
plt.xlabel('$\\nu_d$')
plt.ylabel('$n_d$')
plt.title('Glass Quadruplet Candidates for VNS Chromatic Performance'
                  f', Wavelength {wv_lo} nm to {wv_hi} nm')
plt.grid()
#plt.savefig('OharaHoyaTripletMaterialsForVNS.pdf')

```

```python
gc.save_best_combos(rex)
```

```python
gc.best_gls_combos_df
```

```python
print(gc.weight_per_grp)
print(gc.k_gls_per_grp)
print(gc.num_grp)
```

```python
# Establish downstream glass combination problem
# Give a default buchdahl alpha parameter, which will be overriden by buchdahl_find_alpha(), if run
gd = zg.GlassCombo(wv, i_wv_0, [schott], [2], [0.22], efl=-efl, buchdahl_alpha=2.16, sum_abs_pow_limit=10.0,
                   temp_lo=-10.0, temp_hi=40.0, max_result_rows=3000000, upstream=gc)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
# Now to set up the 5-glass problem, take the first four lines and select
rex.head(6)
```

```python
gr1_lib = gls_lib.select_glasses_usingDataFrame(df=rex.head(6), df_col_names=[('c1','g1'),('c2','g2'),('c3','g3')], inplace=False)
```

<!-- #raw -->
% The raw latex commands in this cell will generate a second reference section, which will work if this
% notebook is downloaded as .tex and compiled as usual (pdflatex, bibtex, pdflatex, pdflatex).
% Some manual editing of the .tex file will be necessary to remove the first reference section or deal with
% other possible minor issues.
\bibliographystyle{unsrt}
\bibliography{biblio}
<!-- #endraw -->

```python
print(gr1_lib)
```

```python
# Establish the glass combination problem, 
# Give a default buchdahl alpha parameter, which will be overriden by buchdahl_find_alpha(), if run
g1 = zg.GlassCombo(wv, i_wv_0, [gr1_lib, gls_lib], [3, 2], [1.0, 1.0], efl=500.0, 
                   buchdahl_alpha=2.21, temp_lo=-10.0, temp_hi=40.0)
```

```python
# Calculate the Buchdahl dispersive power function coefficients eta
g1.buchdahl_fit_eta()
```

```python
# Set up the dask.distributed computing client if not already existing
daclient = g1.dask_client_setup(daclient)
```

```python
# Run the de Albuquerque metrics
%time result1 = g1.dask_run_de_albuquerque(daclient)
```

```python
result1
```

```python
# Make some extracts from the resulting best combinations
rex1 = result1.copy()
# Eliminate combinations with large amounts of (absolute) optical power
rex1 = rex1[rex1['f_1'] < 30.0]
```

```python
# Select for small color aberration
rex1 = rex1[(rex1['f_2'] < 0.01) & (rex1['f_2'] > -0.00) & (rex1['f_2'] != -0.00)]
```

```python
# Select for small opto-thermal sensitivity
# Skip this cell if not required
rex1 = rex1[(rex1['f_4'] < 0.02) & (rex1['f_4'] > -0.02)]
```

```python
rex1.sort_values(by=['f_2'], inplace=True)
display(rex1.head(50))
```

# Extension of Method to Thermo-Chromatic Glass Search




# References

<mark> <b>The bib file biblio.bib was not found

</b> </mark>[<a id="cit-Albuquerque2012" href="#call-Albuquerque2012">Albuquerque2012</a>] !! _This reference was not found in biblio.bib _ !!

[<a id="cit-Wang2019" href="#call-Wang2019">Wang2019</a>] !! _This reference was not found in biblio.bib _ !!

[<a id="cit-Lim2016" href="#call-Lim2016">Lim2016</a>] !! _This reference was not found in biblio.bib _ !!

[<a id="cit-Albuquerque2014" href="#call-Albuquerque2014">Albuquerque2014</a>] !! _This reference was not found in biblio.bib _ !!

[<a id="cit-Albuquerque2016" href="#call-Albuquerque2016">Albuquerque2016</a>] !! _This reference was not found in biblio.bib _ !!

[<a id="cit-Reshidko2013" href="#call-Reshidko2013">Reshidko2013</a>] !! _This reference was not found in biblio.bib _ !!


