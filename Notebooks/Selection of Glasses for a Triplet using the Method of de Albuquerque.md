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

# Application of de Albuquerque Method of Glass Selection
The method of glass selection proposed by de Albuquerque et al. \cite{Albuquerque2012, Albuquerque2014, Albuquerque2016} is illustrated here for searching of 3-glass combinations. Only steps 1 to 3 described in de Albuquerque et al. \cite{Albuquerque2012} are implemented.

The method makes use of the Buchdahl dispersion function where the refractive index of a material, is written as
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

```python
# Major code imports
import numpy as np
import ZemaxGlass as zg
import pandas as pd
%load_ext autoreload
%autoreload 1
%aimport ZemaxGlass
from IPython.core.display import Math, Markdown, Latex
import matplotlib.pyplot as plt
%matplotlib inline
```

## Step 1 - Provide Main System Requirements, Wavelengths and Glass Library

As input data for the method the designer must provide the system effective focal length  $F$ (variable `efl`), the f-number $f$ (`focal_ratio`), the $n$ wavelengths (`wv`) that cover the desired spectral range, and the number of the primary wavelength $\lambda_0$ (`i_wv_0`). A glass library (`gls_lib`) must be provided together with the desired number of glasses in the combination $k$ (`k_gls`).

We will provide four wavelengths in the range 460 nm to 800 nm, one of which will be the primary wavelength.

```python
efl = 400.0  # mm, focal length of the optical design being considered
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
k_gls = 3  # Searching for a 3-glass combination
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
# Calculate opto-thermal coefficients and add to the glass library
gls_lib.add_opto_thermal_coeff(temp_lo, temp_hi, wv_ref=wv_0)
# Filter out materials with extreme opto-thermal coefficients
```

```python
# Compute best fit Buchdahl alpha parameter for the glass library
%time cat, gls, buch_fits = gls_lib.buchdahl_find_alpha(wv, wv_0, show_progress=True)
print(f'Processed {len(gls)} glasses from the {pick_catalog.title()} catalog.')
```

```python
# Calculate and display the mean Buchdahl alpha parameter value
buchdahl_alpha = buch_fits[:, 0].mean()
display(Latex(f'The mean, optimally fitted Buchdahl $\\alpha$ parameter for this glass library is {buchdahl_alpha:4.6f}'))
```

```python
# Now fit again, this time with a constant (optimal mean) alpha for all glasses, should go quicker
%time cat, gls, buch_fit_nu, refr_index_0 = gls_lib.buchdahl_fit(wv, wv_0, alpha=buchdahl_alpha, show_progress=True)
print(f'Processed {len(gls)} glasses.')
```

<!-- #region -->
## Step 2 - Calculate the $\bar{\eta}$ Matrix
The Buchdahl dispersion coefficients $\nu_i$ for a specific glass can be computed by performing a least squares fit to refractive indices calculated from the glass catalog dispersion relation \cite{SchottTIE29}. The fit can be performed allowing $\alpha$ to vary as well as the $\nu_i$ coefficients. However, a fixed Buchdahl $\alpha$ parameter must be used for all glasses and this can be taken as the mean optimised $\alpha$ for a specific set of glasses over a specific wavelength region. Once the best mean $\alpha$ has been determined, the least squares fit for all glasses in the set must be performed again with fixed $\alpha$.

Once the $\nu_i$ coeffcients have been determined for the $k$ glasses in the glass combination to be evaluated, the $\eta_{ij}$ elements of the matrix $\bar{\eta}$ are computed using Equation \ref{eq:eta_i}. These are the coefficients of the Buchdahl dispersive power function for the $k$-glass combination. 

Alternatively, provided that the $\alpha$ parameter is known (computed as indicated above) the $\eta_i$ coefficients for a particular glass can be fitted directly by exploiting Equation \ref{eq:dispersive_power}. This alternative method proceeds by linear algebra as follows.

First form a matrix of the powers of $\omega_i$ as
\begin{equation}\label{eq:omega_bar}
\bar{\omega} = \left[
\begin{array}{cccc}
   \omega_1     & \omega_1^2     & \cdots & \omega_1^{n-1}   \\
   \omega_2     & \omega_2^2     & \cdots & \omega_2^{n-1}   \\
   \vdots       & \vdots         & \ddots & \vdots           \\
   \omega_{n}   & \omega_n^2     & \cdots & \omega_{n}^{n-1} \\
\end{array}
\right]
\end{equation}
The $\bar{\omega}$ matrix is not square being of order $n \times n-1$. However the method requires that the reference wavelength $\lambda_0$ be equal to one of the wavelengths $\lambda_i$. By definition (Equation \ref{eq:omega}), $\omega$ is zero at the reference wavelength, so that one of the rows in $\bar{\omega}$ will be all zeros. This row is deleted to make $\bar{\omega}$ a square matrix of order $n-1 \times n-1$.

For the dispersive power of a specific glass at the chosen wavelengths, Equation \ref{eq:dispersive_power} can be written in matrix form as
\begin{equation}\label{eq:D_bar}
\bar{D} = \bar{\omega} \cdot \bar{\eta}_g,
\end{equation}
where $\bar{D}$, a column vector of $n-1$ elements has the entries $D(\lambda_i)$ (again excluding $D(\lambda_0)=0$) and $\bar{\eta}_g$ are the Buchdahl dispersive power coefficients for the specific glass in question. This is a set of $n-1$ linear equations in $n-1$ variables and easily solved as
\begin{equation}\label{eq:eta_linalg}
\bar{\eta}_g = (\bar{\omega})^{-1} \cdot \bar{D}
\end{equation}

Having selected $k$ (in this exercise $k=3$) for the number of glasses in the combination to be evaluated, the matrix $\bar{\eta}$ can be formed by putting the $\bar{\eta}_g$ for each of the $k$ glasses as colums into the matrix in Equation \ref{eq:eta_bar}.


An intermediate result to be computed to implement the method of de Albuquerque is the square ($n-1 \times n-1$) matrix
\begin{equation}\label{eq:Omega_bar}
\Delta \bar\Omega=\left[
\begin{array}{cccc}
   \omega_1-\omega_2     & \omega_1^2-\omega_2^2      & \cdots & \omega_1^{n-1}-\omega_2^{n-1} \\
   \omega_2-\omega_3     & \omega_2^2-\omega_3^2      & \cdots & \omega_2^{n-1}-\omega_3^{n-1} \\
   \vdots        & \vdots         & \ddots & \vdots    \\
   \omega_{n-1}-\omega_n     & \omega_{n-1}^2-\omega_n^2      & \cdots & \omega_{n-1}^{n-1}-\omega_n^{n-1} \\
\end{array}
\right],
\end{equation}
where the $\omega_i$ Buchdahl spectral coordinates are computed using Equation \ref{eq:omega} at the wavelengths $\lambda_i$. Note that this matrix is a constant for the problem under consideration, depending only on the selected wavelengths $\lambda_i$.

<!-- #endregion -->

```python
# Calculate the eta coefficients for all glasses
eta_gls = buch_fit_nu / (refr_index_0 - 1.0)  # One row per glass in the library
# Calculate the Buchdahl omega coordinates
omega = zg.buchdahl_omega(wv, wv_0, buchdahl_alpha)
display(Latex(f'The Buchdahl $\\omega$ spectral coordinates are {omega}'))
# In this case, the Delta Omega Bar matrix is 3 by 3, so hack this as follows
delta_omega = -np.diff(omega)
delta_omega_2 = -np.diff(omega**2.0)
delta_omega_3 = -np.diff(omega**3.0)
delta_omega_bar = np.column_stack((delta_omega, delta_omega_2, delta_omega_3))
```

```python
# Print out the Delta Omega bar matrix
display(Latex('The $\\Delta \\bar \\Omega$ matrix is'))
print(delta_omega_bar)
```

## Step 3 - Process All Glass Combinations
In this step, all possible combinations of $k$ glasses are processed. If there are $N_g$ glasses in the library (a glass library in this case is defined as a set of glasses that may be selected from the catalogs of one or more glass manufacturers), then the total number of $k$-glass combinations is
\begin{equation}\label{eq:total_gls_comb}
{{N_g}\choose{k}} = \frac{N_g !}{k!(N_g - k)!} 
\end{equation}

It is important to note here that the chromatic aberration correction depends only on the $k$ glass combination and the optical power assigned to each glass. It does not depend on the order in which the glasses are placed in the optical design. The order of the glasses and lens bending (change of lens shape while maintaining optical power constant) are variables the optical designer can exploit to correct other aberrations.  

The normalised optical power $\hat{\phi}_j$ of the three lenses making up each $k$-material combination is computed. The optimal power distribution vector $\hat{\bar \Phi} = (\hat{\phi}_1, \hat{\phi}_2, \hat{\phi}_3)$ to the three lenses is computed using a least squares method as
\begin{equation}\label{eq:Phibarhat}
\hat{\bar \Phi} = \left( \bar{G}^t \cdot \bar{G} \right)^{-1} \cdot \bar{G}^t \cdot \hat{e}
\end{equation}

where $\bar{G}$ is defined as
\begin{equation}\label{eq:Gbar}
\bar G = \left[
\begin{array}{c}
 \bar S \\
  \Delta \bar{\Omega} \cdot  \bar{\eta} \\
\end{array}
\right]
\end{equation}
with $\bar{S}$ a row vector (order $1 \times k$) of ones and $\hat{e}$ a column vector with 1 at the top and zeros below (order $n\times 1$)

Depending on the number of glasses in the library, this step can take quite a long time.

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






```python
# Calculate some vectors for further processing, refer to Albuquerque 2012 for more information
big_s_bar = np.ones(k_gls)  # Row vector of k ones
e_hat = np.vstack((np.array([1.0]), np.zeros((wv.size-1, 1))))  # Column vector with 1 at the top and zeros below
big_n_g = len(gls)  # Number of glasses in the library
# Run through all glass combinations
total_combinations = np.math.factorial(big_n_g) // (np.math.factorial(k_gls) * np.math.factorial(big_n_g - k_gls))
print(f'There are a total of {np.int(total_combinations)} combinations of {k_gls} glasses '
      f'from a library of {big_n_g} glasses to be processed.')
```

```python
# Initialise matrices to hold results
norm_pow = np.zeros((total_combinations, 3))  # Normalised lens power at central wavelength for each glass (k=3)
sum_abs_norm_pow = np.zeros(total_combinations)  # Sum of the absolute normalised power F_1
abs_chroma_pow_delta = np.zeros(total_combinations)  # Modulus of the normalised chromatic power shift F_2 = |CCP|
therm_power_rate = np.zeros(total_combinations)  # Rate of change of total optical power with temperature
delta_temp_F = np.zeros(total_combinations)  # Total absolute focal shift over temperature range delta_temp
delta_color_F = np.zeros(total_combinations)  # Absolute focal shift over wavelength |CCP| * F
delta_F = np.zeros(total_combinations)  # RSS delta_temp_F and delta_color_F
gls_1 = np.zeros(total_combinations, dtype=np.int)  # Record glass numbers for later reference
gls_2 = np.zeros(total_combinations, dtype=np.int)
gls_3 = np.zeros(total_combinations, dtype=np.int)
gls_counter = 0  # Counts through all glass combinations
# The heavy lifting starts here, get a cup of coffee
for i_gls_1 in range(big_n_g):
    # Show a progress bar
    zg.update_progress((gls_counter+1) / total_combinations, bar_length=80)    
    for i_gls_2 in range(i_gls_1 + 1, big_n_g):
        for i_gls_3 in range(i_gls_2 + 1, big_n_g):
            gls_1[gls_counter] = i_gls_1  # Build glass numbers for later lookup
            gls_2[gls_counter] = i_gls_2
            gls_3[gls_counter] = i_gls_3            
            if (i_gls_1 == i_gls_2) or (i_gls_2 == i_gls_3) or (i_gls_3 == i_gls_1):  # want 3 different glasses
                sum_abs_norm_pow[gls_counter] = np.nan  # Only need to set one column to Nan
                gls_counter += 1
                continue  # skip this combination
            # Build the eta bar matrix of Buchdahl dispersive power function coefficients
            # There is one column per glass
            eta_bar = np.column_stack((eta_gls[i_gls_1, :], eta_gls[i_gls_2, :], eta_gls[i_gls_3, :]))
            # Build a vector of opto-thermal coefficients for the three glasses
            gamma_1 = gls_lib.library[cat[i_gls_1]][gls[i_gls_1]]['opto_therm_coeff']
            gamma_2 = gls_lib.library[cat[i_gls_2]][gls[i_gls_2]]['opto_therm_coeff']
            gamma_3 = gls_lib.library[cat[i_gls_3]][gls[i_gls_3]]['opto_therm_coeff']
            gamma_j = np.array([gamma_1, gamma_2, gamma_3])
            # Calculate the G bar matrix 
            big_g_bar = np.vstack((big_s_bar, np.matmul(delta_omega_bar, eta_bar)))
            determinant = np.linalg.det(np.matmul(big_g_bar.T, big_g_bar))
            # Some glasses are very nearly identical, resulting in a bad combination
            # This results in a matrix that cannot be inverted because the determinant is 0
            if determinant == 0.0:
                sum_abs_norm_pow[gls_counter] = np.nan  # Only need to set one column to Nan
                gls_counter += 1
                continue  # skip this combination                
            big_g_bar_mash = np.matmul(np.linalg.inv(np.matmul(big_g_bar.T, big_g_bar)), big_g_bar.T)
            big_phi_barhat = np.matmul(big_g_bar_mash, e_hat)
            # Calculate the chromatic change of power called vector CCP by de Albuquerque
            chroma_power_delta = np.matmul(delta_omega_bar, np.matmul(eta_bar, big_phi_barhat))
            # Calculate the chromatic focal shift by multiplying by the effective focal length
            delta_color_F[gls_counter] = np.linalg.norm(chroma_power_delta) * efl  # |CCP| * F
            # Save the data, chromatic focal shift, power distributions
            norm_pow[gls_counter, :] = big_phi_barhat.T  # Record the normalised power for the 3-glass combo
            # Calculate and save the rate of power change with temperature
            therm_power_rate[gls_counter] = np.abs((big_phi_barhat.T * gamma_j).sum())  # absolute normalised power change
            sum_abs_norm_pow[gls_counter] = np.abs(norm_pow[gls_counter, :]).sum()
            # Focus shift over whole temperature range
            delta_temp_F[gls_counter] = -efl * delta_temp * (big_phi_barhat.T * gamma_j).sum()
            # RSS chromatic and opto-thermal focus shifts
            delta_F[gls_counter] = np.sqrt(delta_temp_F[gls_counter]**2.0 + delta_color_F[gls_counter]**2.0)
            abs_chroma_pow_delta[gls_counter] = np.linalg.norm(chroma_power_delta)  # |CCP|
            gls_counter += 1
# Also give de Albuquerque naming
big_f_1 = sum_abs_norm_pow
big_f_2 = abs_chroma_pow_delta 
# For now, skip the third dimension of the de Albuquerque objective function, which combines some aberration residuals
# And add the fourth dimension of the objective function which is the thermal sensitivity
big_f_4 = delta_temp_F
big_f_5 = delta_F
zg.update_progress(1.0, bar_length=80)
print(f'Processed {gls_counter} combinations of {k_gls} glasses from a library of {big_n_g} glasses')
```

```python
# Build a pandas dataframe
# Get glass names
glname_1 = [gls[i] for i in gls_1]
glname_2 = [gls[i] for i in gls_2]
glname_3 = [gls[i] for i in gls_3]

combo_df = pd.DataFrame({'gls_1': glname_1, 'pow_1': norm_pow[:, 0]/efl, # Give the absolute power
                         'gls_2': glname_2, 'pow_2': norm_pow[:, 1]/efl, 
                         'gls_3': glname_3, 'pow_3': norm_pow[:, 2]/efl,
                         #'df_col': np.abs(chroma_focal_shift).sum(axis=1),
                         'F_1': big_f_1, 'F_2': big_f_2*efl, 'F_4': big_f_4, 'F_5': big_f_5})
# Remove all the rows with any column that is a Nan
combo_df.dropna(how='any', inplace=True)
```

```python
# Eliminate combinations with large amounts of (absolute) optical power
combo_df = combo_df[combo_df['F_1'] < 10.0]
# Select for small color aberration
combo_df = combo_df[combo_df['F_2'] < 0.2]
# Select for small opto-thermal sensitivity
combo_df = combo_df[np.abs(combo_df['F_4']) < 0.2]
```

```python
latex_flag = True
combo_df.sort_values(by=['F_2'], inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Optimal Glass Triplet Candidates for Thermo-Chromatic Performance'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C), Sorted by $F_2$\end{{center}}'))
    display(Latex(combo_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Glass 1', 'Power 1 [$\mathrm{mD}$]',
                                           'Glass 2', 'Power 2 [$\mathrm{mD}$]', 
                                           'Glass 3', 'Power 3 [$\mathrm{mD}$]',
                                           '$F_1$', '$F_2$', '$F_4$ [mm]', '$F_5 [mm]$'],
                                   float_format="%.4f" 
                                  )))
else:
    display(combo_df)
```

```python
combo_df.sort_values(by=['F_5'], inplace=True)
if latex_flag:
    display(Latex('\\clearpage\\begin{center}Optimal Glass Triplet Candidates for Thermo-Chromatic Performance'
                  f' ({temp_lo}$^\circ$C to {temp_hi}$^\circ$C), Sorted by $F_5$\end{{center}}'))
    display(Latex(combo_df.to_latex(index=False, longtable=True, escape=False,
                                   header=['Glass 1', 'Power 1 [$\mathrm{mD}$]',
                                           'Glass 2', 'Power 2 [$\mathrm{mD}$]', 
                                           'Glass 3', 'Power 3 [$\mathrm{mD}$]',
                                           '$F_1$', '$F_2$', '$F_4$ [mm]', '$F_5$ [mm]'],
                                   float_format="%.4f" 
                                  )))
else:
    display(combo_df)
```

## Step 4 - Determination of Aplanatic Solution
The de Albuquerque method proceeds to solve for an assignment of lens surface curvatures (equvalently the radii of curvature) that result in an aplanatic lens solution. An aplanatic solution is free of 3rd order monochromatic spherical aberration and coma. It was known in advance that the problem to be solved would require the use of aspherical surfaces in order to reduce weight. Since the aberration analysis process from here would likely be inapplicable to aspheric solutions, the process was not pursued beyond step 3.

## Step 5 - Computation of 5th Order Spherical and Sphero-Chromatic Aberrations
The best aplanatic realization for the lens is analyzed for higher order aberrations, specifically 5th order normalized spherical aberration and normalized sphero-chromatism. The $F_3$ metric is based on a weighted summation of the wave coefficients for these two aberrations.

## Steps 6 to 8 - Tabulation, Pareto Ranking and Post-Pareto Analysis
The solutions are tabulated and subject to ranking and post-Pareto analysis. The best solution is selected from a table sorted by the metric(s) of interest and the ranking.


# References

[<a id="cit-Albuquerque2012" href="#call-Albuquerque2012">Albuquerque2012</a>] Fonseca Br\'{a}ulio, Sasian Jose, Luis Fabiano <em>et al.</em>, ``_Method of glass selection for color correction in optical system design_'', Optics Express, vol. 20, number 13, pp. 13592--13611, Jun 2012.  [online](http://www.opticsexpress.org/abstract.cfm?URI=oe-20-13-13592)

[<a id="cit-SchottTIE29" href="#call-SchottTIE29">SchottTIE29</a>] {Schott Advanced, ``TIE-29 Refractive Index and Dispersion'', Schott Inc., number: ,   2016.  [online](https://www.schott.com/d/advanced_optics/02ffdb0d-00a6-408f-84a5-19de56652849/1.2/tie_29_refractive_index_and_dispersion_eng.pdf)

[<a id="cit-Reshidko2013" href="#call-Reshidko2013">Reshidko2013</a>] D. Reshidko and J. Sasián, ``_Method of calculation and tables of optothermal coefficients and thermal diffusivities for glass_'', Optical System Alignment, Tolerancing, and Verification VII,  2013.  [online](https://doi.org/10.1117/12.2036112)

[<a id="cit-de2014multi" href="#call-de2014multi">de2014multi</a>] !! _This reference was not found in biblio.bib _ !!

[<a id="cit-Albuquerque2014" href="#call-Albuquerque2014">Albuquerque2014</a>] B. F. C. de Albuquerque, ``_A multi-objective memetic approach for the automatic design of optical systems_'',  2014.  [online](http://mtc-m16d.sid.inpe.br/col/sid.inpe.br/mtc-m19/2014/01.15.12.56/doc/publicacao.pdf)

[<a id="cit-Albuquerque2016" href="#call-Albuquerque2016">Albuquerque2016</a>] Fonseca Br\'{a}ulio, Luis Fabiano and Silva Amauri, ``_Multi-objective approach for the automatic design of optical systems_'', Opt. Express, vol. 24, number 6, pp. 6619--6643, Mar 2016.  [online](http://www.opticsexpress.org/abstract.cfm?URI=oe-24-6-6619)



<!-- #raw -->
% The raw latex commands in this cell will generate a second reference section, which will work if this
% notebook is downloaded as .tex and compiled as usual (pdflatex, bibtex, pdflatex, pdflatex).
% Some manual editing of the .tex file will be necessary to remove the first reference section or deal with
% other possible minor issues.
\bibliographystyle{unsrt}
\bibliography{biblio}
<!-- #endraw -->
