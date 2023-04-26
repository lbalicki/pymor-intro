---
celltoolbar: Slideshow
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.8.13
rise:
  header: <img class="pymor-logo" src="logo/pymor_logo.png" alt="pyMOR logo" width=12%
    />
  scroll: true
---

+++ {"slideshow": {"slide_type": "slide"}}

# pyMOR Intro

+++ {"slideshow": {"slide_type": "subslide"}}

## Why?

## What?

## How?

+++ {"slideshow": {"slide_type": "slide"}}

# Why?

+++ {"slideshow": {"slide_type": "subslide"}}

## Why FLOSS?

+++ {"slideshow": {"slide_type": "fragment"}}

1. Why use FLOSS?
1. Why publish FLOSS?
1. Why contribute to a FLOSS project?

+++ {"slideshow": {"slide_type": "subslide"}}

## Why Python?

+++ {"slideshow": {"slide_type": "subslide"}}

## Why pyMOR?

+++ {"slideshow": {"slide_type": "fragment"}}

1. Why was pyMOR created?

+++ {"slideshow": {"slide_type": "fragment"}}

<table width=70%>
    <tr>
        <td><img src="figs/renefritze.jpg"></td>
        <td><img src="figs/sdrave.jpg"></td>
        <td><img src="figs/ftschindler.jpg"></td>
    </tr>
    <tr>
        <td>René Fritze</td>
        <td>Stephan Rave</td>
        <td>Felix Schindler</td>
    </tr>
</table>

+++ {"slideshow": {"slide_type": "subslide"}}

<center><img src="figs/multibat_project.jpg" width=50%></center>

+++ {"slideshow": {"slide_type": "subslide"}}

<center><img src="figs/multibat_interfaces.jpg" width=60%></center>

+++ {"slideshow": {"slide_type": "subslide"}}

2. Why should I use pyMOR?

+++ {"slideshow": {"slide_type": "slide"}}

# What?

+++ {"slideshow": {"slide_type": "subslide"}}

## What is pyMOR?

- Python package for model order reduction
- started in October 2012
- 25k lines of Python code (17k lines of docs)
- 8k commits
- permissive license (BSD 2-clause)

+++ {"slideshow": {"slide_type": "fragment"}}

## Goals

1. one library for algorithm development and large-scale applications
2. unified approach on MOR

+++ {"slideshow": {"slide_type": "subslide"}}

## Main Developers

<table width=70%>
    <tr>
        <td><img src="figs/lbalicki.jpg"></td>
        <td><img src="figs/renefritze.jpg"></td>
        <td><img src="figs/HenKlei.jpg"></td>
        <td><img src="figs/pmli.jpg"></td>
        <td><img src="figs/sdrave.jpg"></td>
        <td><img src="figs/ftschindler.jpg"></td>
    </tr>
    <tr>
        <td>Linus Balicki</td>
        <td>René Fritze</td>
        <td>Hendrik Kleikamp</td>
        <td>Petar Mlinarić</td>
        <td>Stephan Rave</td>
        <td>Felix Schindler</td>
    </tr>
</table>

+++ {"slideshow": {"slide_type": "fragment"}}

## Code Contributors

<table width=90%>
    <tr>
        <td><img src="figs/meretp.jpg"></td>
        <td><img src="figs/bergdola.jpg"></td>
        <td><img src="figs/pbuchfink.jpg"></td>
        <td><img src="figs/andreasbuhr.jpg"></td>
        <td><img src="figs/cabuze.jpg"></td>
        <td><img src="figs/mdessole.jpg"></td>
        <td><img src="figs/deneick.jpg"></td>
    </tr>
    <tr>
        <td>Meret Behrens</td>
        <td>@bergdola</td>
        <td>Patrick Buchfink</td>
        <td>Andreas Buhr</td>
        <td>@cabuze</td>
        <td>Monica Dessole</td>
        <td>Dennis Eickhorn</td>
    </tr>
    <tr>
        <td><img src="figs/hvhue.jpg"></td>
        <td><img src="figs/TiKeil.jpg"></td>
        <td><img src="figs/michaellaier.jpg"></td>
        <td><img src="figs/JuliaBru.jpg"></td>
        <td><img src="figs/gdmcbain.jpg"></td>
        <td><img src="figs/mechiluca.jpg"></td>
        <td><img src="figs/MohamedAdelNaguib.jpg"></td>
    </tr>
    <tr>
        <td>@hvhue</td>
        <td>Tim Keil</td>
        <td>Michael Laier</td>
        <td>Julia Maiwald</td>
        <td>G. D. McBain</td>
        <td>Luca Mechelli</td>
        <td>Mohamed Adel Naguib Ahmed</td>
    </tr>
    <tr>
        <td><img src="figs/Jonas-Nicodemus.jpg"></td>
        <td><img src="figs/peoe.jpg"></td>
        <td><img src="figs/MagnusOstertag.jpg"></td>
        <td><img src="figs/artpelling.jpg"></td>
        <td><img src="figs/michaelschaefer.jpg"></td>
        <td><img src="figs/ullmannsven.jpg"></td>
        <td><img src="figs/josefinez.jpg"></td>
    </tr>
    <tr>
        <td>Jonas Nicodemus</td>
        <td>Peter Oehme</td>
        <td>Magnus Ostertag</td>
        <td>Art Pelling</td>
        <td>Michael Schaefer</td>
        <td>Sven Ullmann</td>
        <td>Josefine Zeller</td>
    </tr>
</table>

+++ {"slideshow": {"slide_type": "slide"}}

w# How?

+++ {"slideshow": {"slide_type": "subslide"}}

## LTI Models

+++ {"slideshow": {"slide_type": "-"}}

### Imports and Settings

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pymor.core.logger import set_log_levels
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
plt.rcParams['axes.grid'] = True
set_log_levels({
    'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
    'pymor.reductors.basic.LTIPGReductor': 'WARNING',
})
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Loading from Files

Rail model from [MOR Wiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Steel_Profile).

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
from pymor.models.iosys import LTIModel

rail_fom = LTIModel.from_abcde_files('data/rail/rail_5177_c60')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rail_fom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rail_fom.A
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rail_fom.A.matrix
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(rail_fom)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### FOM Magnitude Plot

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
w = (1e-7, 1e3)
_ = rail_fom.transfer_function.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Hankel Singular Values

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rail_hsv = rail_fom.hsv()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fig, ax = plt.subplots()
_ = ax.semilogy(rail_hsv, '.')
_ = ax.set_title('Hankel singular values')
```

+++ {"slideshow": {"slide_type": "slide"}}

## Balanced Truncation

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.reductors.bt import BTReductor

rail_bt = BTReductor(rail_fom)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rail_rom_bt = rail_bt.reduce(20)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rail_rom_bt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
rail_rom_bt.A.matrix
```

+++ {"slideshow": {"slide_type": "subslide"}}

### BT Poles

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fig, ax = plt.subplots()
poles = rail_rom_bt.poles()
_ = ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('BT poles')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
poles.real.max()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### FOM and BT Magnitude Plots

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
w = (1e-6, 1e3)
_ = rail_fom.transfer_function.mag_plot(w)
_ = rail_rom_bt.transfer_function.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### BT Error System

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rail_err_bt = rail_fom - rail_rom_bt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = rail_err_bt.transfer_function.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### $\mathcal{H}_2$ Relative Error

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rail_err_bt.h2_norm() / rail_fom.h2_norm()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### $\mathcal{H}_\infty$ Error Bounds

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = plt.semilogy(rail_bt.error_bounds(), '.-')
```

+++ {"slideshow": {"slide_type": "slide"}}

## IRKA

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.reductors.h2 import IRKAReductor

rail_irka = IRKAReductor(rail_fom)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rail_rom_irka = rail_irka.reduce(20, conv_crit='h2', num_prev=10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
rail_rom_irka
```

+++ {"slideshow": {"slide_type": "subslide"}}

### IRKA Convergence

```{code-cell} ipython3
_ = plt.semilogy(rail_irka.conv_crit, '.-')
```

+++ {"slideshow": {"slide_type": "subslide"}}

### IRKA Poles

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fig, ax = plt.subplots()
poles = rail_rom_irka.poles()
_ = ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('IRKA poles')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
poles.real.max()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### FOM and IRKA Magnitude Plots

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
w = (1e-6, 1e3)
_ = rail_fom.transfer_function.mag_plot(w)
_ = rail_rom_irka.transfer_function.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### IRKA Error System

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rail_err_irka = rail_fom - rail_rom_irka
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig, ax = plt.subplots()
_ = rail_err_bt.transfer_function.mag_plot(w, ax=ax, label='BT')
_ = rail_err_irka.transfer_function.mag_plot(w, ax=ax, label='IRKA')
_ = ax.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### $\mathcal{H}_2$ Relative Error

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rail_err_irka.h2_norm() / rail_fom.h2_norm()
```

+++ {"slideshow": {"slide_type": "subslide"}}

## Transfer Function

+++ {"slideshow": {"slide_type": "fragment"}}

Heat equation over a semi-infinite rod from \[Beattie/Gugercin '12\].

$$
\begin{align*}
  H(s) & = e^{-\sqrt{s}} \\
  H'(s) & = -\frac{e^{-\sqrt{s}}}{2 \sqrt{s}}
\end{align*}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.models.transfer_function import TransferFunction

tf = TransferFunction(
    1,
    1,
    lambda s: np.array([[np.exp(-np.sqrt(s))]]),
    dtf=lambda s: np.array([[-np.exp(-np.sqrt(s)) / (2 * np.sqrt(s))]]),
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
tf
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Bode Plot

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
w_tf = (1e-7, 1e4)
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, squeeze=False, constrained_layout=True)
_ = tf.bode_plot(w_tf, ax=ax)
```

+++ {"slideshow": {"slide_type": "slide"}}

## TF-IRKA

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.reductors.h2 import TFIRKAReductor

tf_irka = TFIRKAReductor(tf)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
tf_rom = tf_irka.reduce(20)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
tf_rom
```

+++ {"slideshow": {"slide_type": "subslide"}}

### TF-IRKA Poles

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fig, ax = plt.subplots()
poles = tf_rom.poles()
_ = ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('IRKA poles')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
poles.real.max()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Bode Plots

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, squeeze=False, constrained_layout=True)
_ = tf.bode_plot(w_tf, ax=ax)
_ = tf_rom.transfer_function.bode_plot(w_tf, ax=ax)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Error System

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
tf_err = tf - tf_rom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = tf_err.mag_plot(w_tf)
```

+++ {"slideshow": {"slide_type": "slide"}}

## AAA

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.reductors.aaa import PAAAReductor

sampling_values = 1j * np.logspace(-4, 2, 100)
sampling_values = np.concatenate((sampling_values, -sampling_values))
sampling_values = [sampling_values]
aaa = PAAAReductor(sampling_values, tf)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
aaa_rom = aaa.reduce(tol=1e-4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
aaa_rom
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Bode Plots

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fig, ax = plt.subplots(2, 1, squeeze=False, constrained_layout=True)
_ = tf.bode_plot(w_tf, ax=ax)
_ = tf_rom.transfer_function.bode_plot(w_tf, ax=ax)
_ = aaa_rom.bode_plot(w_tf, ax=ax)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Error System

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
aaa_err = tf - aaa_rom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig, ax = plt.subplots()
_ = tf_err.mag_plot(w_tf, ax=ax, label='TF-IRKA')
_ = aaa_err.mag_plot(w_tf, ax=ax, label='AAA')
_ = ax.legend()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Parametric LTI Models

+++ {"slideshow": {"slide_type": "fragment"}}

Cookie model (thermal block) example from [MOR Wiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Thermal_Block).

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import scipy.io as spio

mat = spio.loadmat('data/cookie/ABCE.mat')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
mat.keys()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
A0 = mat['A0']
A1 = 0.2 * mat['A1'] + 0.4 * mat['A2'] + 0.6 * mat['A3'] + 0.8 * mat['A4']
B = mat['B']
C = mat['C']
E = mat['E']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
A0
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.operators.numpy import NumpyMatrixOperator

A0op = NumpyMatrixOperator(A0)
A1op = NumpyMatrixOperator(A1)
Bop = NumpyMatrixOperator(B)
Cop = NumpyMatrixOperator(C)
Eop = NumpyMatrixOperator(E)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
A0op
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.parameters.functionals import ProjectionParameterFunctional

Aop = A0op + ProjectionParameterFunctional('p') * A1op
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
Aop
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
cookie_fom = LTIModel(Aop, Bop, Cop, E=Eop)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
cookie_fom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
cookie_fom.parameters
```

+++ {"slideshow": {"slide_type": "subslide"}}

## Magnitude Plot

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
num_w = 10
num_p = 10
ws = np.logspace(-4, 4, num_w)
ps = np.logspace(-6, 2, num_p)
Hwp = np.empty((num_p, num_w))
for i in range(num_p):
    Hwp[i] = spla.norm(cookie_fom.transfer_function.freq_resp(ws, mu=ps[i]), axis=(1, 2))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig, ax = plt.subplots()
out = ax.pcolormesh(ws, ps, Hwp, shading='gouraud', norm=LogNorm())
ax.set(
    xscale='log',
    yscale='log',
    xlabel=r'Frequency $\omega$ (rad/s)',
    ylabel='Parameter $p$',
    title=r'$\Vert H(i \omega, p) \Vert$',
)
ax.grid(False)
_ = fig.colorbar(out)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Interpolation

```{code-cell} ipython3
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.reductors.interpolation import LTIBHIReductor

s_samples = np.logspace(-1, 1, 5)
s_samples = np.concatenate((1j * s_samples, -1j * s_samples))
p_samples = np.logspace(-3, -1, 5)
V = cookie_fom.A.source.empty()
W = cookie_fom.A.source.empty()
for p in p_samples:
    interp = LTIBHIReductor(cookie_fom, mu=p)
    interp.reduce(s_samples, np.ones((len(s_samples), 1)), np.ones((len(s_samples), 4)))
    V.append(interp.V)
    W.append(interp.W)

_ = gram_schmidt(V, copy=False)
_ = gram_schmidt(W, copy=False)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
V
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.basic import LTIPGReductor

pg = LTIPGReductor(cookie_fom, W, V)
cookie_rom = pg.reduce()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
cookie_rom
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Error System

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cookie_err = cookie_fom - cookie_rom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
Hwp_err = np.empty((num_p, num_w))
for i in range(num_p):
    Hwp_err[i] = spla.norm(cookie_err.transfer_function.freq_resp(ws, mu=ps[i]), axis=(1, 2))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig, ax = plt.subplots()
out = ax.pcolormesh(ws, ps, Hwp_err, shading='gouraud', norm=LogNorm())
ax.set(
    xscale='log',
    yscale='log',
    xlabel=r'Frequency $\omega$ (rad/s)',
    ylabel='Parameter $p$',
    title=r'$\Vert H(i \omega, p) - H_r(i \omega, p) \Vert$',
)
ax.grid(False)
_ = fig.colorbar(out)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### ROM Poles

```{code-cell} ipython3
for p in ps:
    poles = cookie_rom.poles(mu=p)
    print(poles.real.max())
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Galerkin Projection

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
from pymor.algorithms.pod import pod

VW = V.copy()
VW.append(W)
VW, svals = pod(VW, modes=50)
```

```{code-cell} ipython3
VW
```

```{code-cell} ipython3
galerkin = LTIPGReductor(cookie_fom, VW, VW)
cookie_rom_g = galerkin.reduce()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Error System 2

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cookie_err2 = cookie_fom - cookie_rom_g
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
Hwp_err2 = np.empty((num_p, num_w))
for i in range(num_p):
    Hwp_err2[i] = spla.norm(cookie_err2.transfer_function.freq_resp(ws, mu=ps[i]), axis=(1, 2))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig, ax = plt.subplots()
out = ax.pcolormesh(ws, ps, Hwp_err2, shading='gouraud', norm=LogNorm())
ax.set(
    xscale='log',
    yscale='log',
    xlabel=r'Frequency $\omega$ (rad/s)',
    ylabel='Parameter $p$',
    title=r'$\Vert H(i \omega, p) - H_r(i \omega, p) \Vert$',
)
ax.grid(False)
_ = fig.colorbar(out)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### ROM Poles 2

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
for p in ps:
    poles = cookie_rom_g.poles(mu=p)
    print(poles.real.max())
```

+++ {"slideshow": {"slide_type": "slide"}}

## Nonlinear MOR

+++

Nonlinear RC example from [MOR Wiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Nonlinear_RC_Ladder) (Model 1).

$$
\begin{align*}
  \dot{x}(t) & = A x(t) - g(A_0 x(t)) + g(A_1 x(t)) - g(A_2 x(t)) + B u(t) \\
  y(t) & = C x(t)
\end{align*}
$$

```{code-cell} ipython3
import scipy.sparse as sps
from pymor.models.basic import InstationaryModel
```

```{code-cell} ipython3
n = 100
A_rc = sps.diags([(n - 1) * [1], n * [-2], (n - 1) * [1]], [-1, 0, 1], format='csc')
B_rc = np.zeros((n, 1))
B_rc[0, 0] = 1
C_rc = B_rc.T
```

```{code-cell} ipython3
A_rc.toarray()
```

```{code-cell} ipython3
A0_rc = sps.lil_matrix((n, n))
A0_rc[0, 0] = 1
A0_rc = A0_rc.tocsc()

A1_rc = sps.diags([(n - 1) * [1], n * [-1]], [-1, 0], format='lil')
A1_rc[0, 0] = 0
A1_rc = A1_rc.tocsc()

A2_rc = sps.diags([n * [1], (n - 1) * [-1]], [0, 1], format='lil')
A2_rc[-1, -1] = 0
A2_rc = A2_rc.tocsc()
```

```{code-cell} ipython3
A0_rc.toarray()
```

```{code-cell} ipython3
A1_rc.toarray()
```

```{code-cell} ipython3
A2_rc.toarray()
```

```{code-cell} ipython3
g = lambda x: np.exp(40 * x) - 1
```

```{code-cell} ipython3
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace

class ComponentwiseOperator(Operator):
    def __init__(self, mapping, dim_source=1, dim_range=1, linear=False,
                 source_id=None, range_id=None, solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(dim_source, source_id)
        self.range = NumpyVectorSpace(dim_range, range_id)

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        return self.range.make_array(self.mapping(U.to_numpy()))

    def restricted(self, dofs):
        return self.with_(dim_source=len(dofs), dim_range=len(dofs)), dofs
```

```{code-cell} ipython3
g_op = ComponentwiseOperator(g, dim_source=n, dim_range=n)

A_rc_op = NumpyMatrixOperator(A_rc)
A0_rc_op = NumpyMatrixOperator(A0_rc)
A1_rc_op = NumpyMatrixOperator(A1_rc)
A2_rc_op = NumpyMatrixOperator(A2_rc)
B_rc_op = NumpyMatrixOperator(B_rc)
C_rc_op = NumpyMatrixOperator(C_rc)
```

```{code-cell} ipython3
from pymor.operators.constructions import LinearInputOperator
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
```

```{code-cell} ipython3
T = 1
x0 = A_rc_op.source.zeros(1)
operator = -A_rc_op + g_op @ A0_rc_op - g_op @ A1_rc_op + g_op @ A2_rc_op
rhs = LinearInputOperator(B_rc_op)
```

```{code-cell} ipython3
rc_fom = InstationaryModel(T, x0, operator, rhs,
                           time_stepper=ExplicitEulerTimeStepper(100),
                           output_functional=C_rc_op)
```

```{code-cell} ipython3
rc_output = rc_fom.output(input=1)
```

```{code-cell} ipython3
_ = plt.plot(np.linspace(0, T, 101), rc_output)
```

```{code-cell} ipython3
X_rc = rc_fom.solve(input=1)
```

```{code-cell} ipython3
pod_vec, pod_val = pod(X_rc)
```

```{code-cell} ipython3
_ = plt.semilogy(pod_val, '.-')
```

```{code-cell} ipython3
from pymor.reductors.basic import InstationaryRBReductor
```

```{code-cell} ipython3
rb = InstationaryRBReductor(rc_fom, pod_vec)
rc_rom = rb.reduce()
```
