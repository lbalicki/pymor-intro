# ---
# jupyter:
#   celltoolbar: Slideshow
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.13
#   rise:
#     header: <img class="pymor-logo" src="logo/pymor_logo.png" alt="pyMOR logo" width=12%
#       />
#     scroll: true
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # pyMOR Intro

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Why?
#
# ## What?
#
# ## How?

# %% [markdown] slideshow={"slide_type": "subslide"}
# Why?

# %% [markdown] slideshow={"slide_type": "fragment"}
# ## Why FLOSS?

# %% [markdown] slideshow={"slide_type": "fragment"}
# 1. Why use FLOSS?
# 1. Why publish FLOSS?
# 1. Why contribute to a FLOSS project?

# %% [markdown] slideshow={"slide_type": "fragment"}
# ## Why Python?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Why pyMOR?

# %% [markdown] slideshow={"slide_type": "fragment"}
# 1. Why was pyMOR created?

# %% [markdown] slideshow={"slide_type": "fragment"}
# <table width=70%>
#     <tr>
#         <td><img src="figs/renefritze.jpg"></td>
#         <td><img src="figs/sdrave.jpg"></td>
#         <td><img src="figs/ftschindler.jpg"></td>
#     </tr>
#     <tr>
#         <td>René Fritze</td>
#         <td>Stephan Rave</td>
#         <td>Felix Schindler</td>
#     </tr>
# </table>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <center><img src="figs/multibat_project.jpg" width=50%></center>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <center><img src="figs/multibat_interfaces.jpg" width=60%></center>

# %% [markdown] slideshow={"slide_type": "subslide"}
# 2. Why should I use pyMOR?

# %% [markdown] slideshow={"slide_type": "slide"}
# What?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## What is pyMOR?
#
# - Python package for model order reduction
# - started in October 2012
# - 25k lines of Python code (17k lines of docs)
# - 8k commits
# - permissive license (BSD 2-clause)

# %% [markdown] slideshow={"slide_type": "fragment"}
# ## Goals
#
# 1. one library for algorithm development and large-scale applications
# 2. unified approach on MOR

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Main Developers
#
# <table width=70%>
#     <tr>
#         <td><img src="figs/lbalicki.jpg"></td>
#         <td><img src="figs/renefritze.jpg"></td>
#         <td><img src="figs/HenKlei.jpg"></td>
#         <td><img src="figs/pmli.jpg"></td>
#         <td><img src="figs/sdrave.jpg"></td>
#         <td><img src="figs/ftschindler.jpg"></td>
#     </tr>
#     <tr>
#         <td>Linus Balicki</td>
#         <td>René Fritze</td>
#         <td>Hendrik Kleikamp</td>
#         <td>Petar Mlinarić</td>
#         <td>Stephan Rave</td>
#         <td>Felix Schindler</td>
#     </tr>
# </table>

# %% [markdown] slideshow={"slide_type": "fragment"}
# ## Code Contributors
#
# <table width=90%>
#     <tr>
#         <td><img src="figs/meretp.jpg"></td>
#         <td><img src="figs/bergdola.jpg"></td>
#         <td><img src="figs/pbuchfink.jpg"></td>
#         <td><img src="figs/andreasbuhr.jpg"></td>
#         <td><img src="figs/cabuze.jpg"></td>
#         <td><img src="figs/mdessole.jpg"></td>
#         <td><img src="figs/deneick.jpg"></td>
#     </tr>
#     <tr>
#         <td>Meret Behrens</td>
#         <td>@bergdola</td>
#         <td>Patrick Buchfink</td>
#         <td>Andreas Buhr</td>
#         <td>@cabuze</td>
#         <td>Monica Dessole</td>
#         <td>Dennis Eickhorn</td>
#     </tr>
#     <tr>
#         <td><img src="figs/hvhue.jpg"></td>
#         <td><img src="figs/TiKeil.jpg"></td>
#         <td><img src="figs/michaellaier.jpg"></td>
#         <td><img src="figs/JuliaBru.jpg"></td>
#         <td><img src="figs/gdmcbain.jpg"></td>
#         <td><img src="figs/mechiluca.jpg"></td>
#         <td><img src="figs/MohamedAdelNaguib.jpg"></td>
#     </tr>
#     <tr>
#         <td>@hvhue</td>
#         <td>Tim Keil</td>
#         <td>Michael Laier</td>
#         <td>Julia Maiwald</td>
#         <td>G. D. McBain</td>
#         <td>Luca Mechelli</td>
#         <td>Mohamed Adel Naguib Ahmed</td>
#     </tr>
#     <tr>
#         <td><img src="figs/Jonas-Nicodemus.jpg"></td>
#         <td><img src="figs/peoe.jpg"></td>
#         <td><img src="figs/MagnusOstertag.jpg"></td>
#         <td><img src="figs/artpelling.jpg"></td>
#         <td><img src="figs/michaelschaefer.jpg"></td>
#         <td><img src="figs/ullmannsven.jpg"></td>
#         <td><img src="figs/josefinez.jpg"></td>
#     </tr>
#     <tr>
#         <td>Jonas Nicodemus</td>
#         <td>Peter Oehme</td>
#         <td>Magnus Ostertag</td>
#         <td>Art Pelling</td>
#         <td>Michael Schaefer</td>
#         <td>Sven Ullmann</td>
#         <td>Josefine Zeller</td>
#     </tr>
# </table>

# %% [markdown] slideshow={"slide_type": "slide"}
# How?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## LTI Models

# %% [markdown] slideshow={"slide_type": "fragment"}
# ### Imports and Settings

# %% slideshow={"slide_type": "-"}
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from pymor.core.logger import set_log_levels

# %% slideshow={"slide_type": "-"}
plt.rcParams['axes.grid'] = True
set_log_levels({
    'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
    'pymor.reductors.basic.LTIPGReductor': 'WARNING',
})

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Loading from Files
#
# Rail model from [MOR Wiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Steel_Profile).

# %% slideshow={"slide_type": "-"}
from pymor.models.iosys import LTIModel

rail_fom = LTIModel.from_abcde_files('data/rail/rail_5177_c60')

# %% slideshow={"slide_type": "fragment"}
rail_fom

# %% slideshow={"slide_type": "fragment"}
rail_fom.A

# %% slideshow={"slide_type": "fragment"}
rail_fom.A.matrix

# %% slideshow={"slide_type": "fragment"}
print(rail_fom)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### FOM Magnitude Plot

# %% slideshow={"slide_type": "-"}
w = (1e-7, 1e3)
_ = rail_fom.transfer_function.mag_plot(w)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Hankel Singular Values

# %% slideshow={"slide_type": "-"}
rail_hsv = rail_fom.hsv()

# %% slideshow={"slide_type": "fragment"}
fig, ax = plt.subplots()
_ = ax.semilogy(rail_hsv, '.')
_ = ax.set_title('Hankel singular values')

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Balanced Truncation

# %% slideshow={"slide_type": "fragment"}
from pymor.reductors.bt import BTReductor

rail_bt = BTReductor(rail_fom)

# %% slideshow={"slide_type": "fragment"}
rail_rom_bt = rail_bt.reduce(20)

# %% slideshow={"slide_type": "fragment"}
rail_rom_bt

# %% slideshow={"slide_type": "fragment"}
rail_rom_bt.A.matrix

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### BT Poles

# %% slideshow={"slide_type": "-"}
fig, ax = plt.subplots()
poles = rail_rom_bt.poles()
_ = ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('BT poles')

# %% slideshow={"slide_type": "fragment"}
poles.real.max()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### FOM and BT Magnitude Plots

# %% slideshow={"slide_type": "-"}
w = (1e-6, 1e3)
_ = rail_fom.transfer_function.mag_plot(w)
_ = rail_rom_bt.transfer_function.mag_plot(w)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### BT Error System

# %% slideshow={"slide_type": "-"}
rail_err_bt = rail_fom - rail_rom_bt

# %% slideshow={"slide_type": "fragment"}
_ = rail_err_bt.transfer_function.mag_plot(w)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### $\mathcal{H}_2$ Relative Error

# %% slideshow={"slide_type": "-"}
rail_err_bt.h2_norm() / rail_fom.h2_norm()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### $\mathcal{H}_\infty$ Error Bounds

# %% slideshow={"slide_type": "-"}
_ = plt.semilogy(rail_bt.error_bounds(), '.-')

# %% [markdown] slideshow={"slide_type": "slide"}
# ## IRKA

# %% slideshow={"slide_type": "fragment"}
from pymor.reductors.h2 import IRKAReductor

rail_irka = IRKAReductor(rail_fom)

# %% slideshow={"slide_type": "fragment"}
rail_rom_irka = rail_irka.reduce(20, conv_crit='h2', num_prev=10)

# %% slideshow={"slide_type": "fragment"}
rail_rom_irka

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### IRKA Convergence

# %%
_ = plt.semilogy(rail_irka.conv_crit, '.-')

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### IRKA Poles

# %% slideshow={"slide_type": "-"}
fig, ax = plt.subplots()
poles = rail_rom_irka.poles()
_ = ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('IRKA poles')

# %% slideshow={"slide_type": "fragment"}
poles.real.max()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### FOM and IRKA Magnitude Plots

# %% slideshow={"slide_type": "-"}
w = (1e-6, 1e3)
_ = rail_fom.transfer_function.mag_plot(w)
_ = rail_rom_irka.transfer_function.mag_plot(w)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### IRKA Error System

# %% slideshow={"slide_type": "-"}
rail_err_irka = rail_fom - rail_rom_irka

# %% slideshow={"slide_type": "fragment"}
fig, ax = plt.subplots()
_ = rail_err_bt.transfer_function.mag_plot(w, ax=ax, label='BT')
_ = rail_err_irka.transfer_function.mag_plot(w, ax=ax, label='IRKA')
_ = ax.legend()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### $\mathcal{H}_2$ Relative Error

# %% slideshow={"slide_type": "-"}
rail_err_irka.h2_norm() / rail_fom.h2_norm()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Transfer Function

# %% [markdown] slideshow={"slide_type": "fragment"}
# Heat equation over a semi-infinite rod from \[Beattie/Gugercin '12\].
#
# $$
# \begin{align*}
#   H(s) & = e^{-\sqrt{s}} \\
#   H'(s) & = -\frac{e^{-\sqrt{s}}}{2 \sqrt{s}}
# \end{align*}
# $$

# %% slideshow={"slide_type": "fragment"}
from pymor.models.transfer_function import TransferFunction

tf = TransferFunction(
    1,
    1,
    lambda s: np.array([[np.exp(-np.sqrt(s))]]),
    dtf=lambda s: np.array([[-np.exp(-np.sqrt(s)) / (2 * np.sqrt(s))]]),
)

# %% slideshow={"slide_type": "fragment"}
tf

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Bode Plot

# %% slideshow={"slide_type": "-"}
w_tf = (1e-7, 1e4)
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, squeeze=False, constrained_layout=True)
_ = tf.bode_plot(w_tf, ax=ax)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## TF-IRKA

# %% slideshow={"slide_type": "fragment"}
from pymor.reductors.h2 import TFIRKAReductor

tf_irka = TFIRKAReductor(tf)

# %% slideshow={"slide_type": "fragment"}
tf_rom = tf_irka.reduce(20)

# %% slideshow={"slide_type": "fragment"}
tf_rom

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### TF-IRKA Poles

# %% slideshow={"slide_type": "-"}
fig, ax = plt.subplots()
poles = tf_rom.poles()
_ = ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('IRKA poles')

# %% slideshow={"slide_type": "fragment"}
poles.real.max()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Bode Plots

# %% slideshow={"slide_type": "-"}
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, squeeze=False, constrained_layout=True)
_ = tf.bode_plot(w_tf, ax=ax)
_ = tf_rom.transfer_function.bode_plot(w_tf, ax=ax)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Error System

# %% slideshow={"slide_type": "-"}
tf_err = tf - tf_rom

# %% slideshow={"slide_type": "fragment"}
_ = tf_err.mag_plot(w_tf)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## AAA

# %% slideshow={"slide_type": "fragment"}
from pymor.reductors.aaa import PAAAReductor

sampling_values = 1j * np.logspace(-4, 2, 100)
sampling_values = np.concatenate((sampling_values, -sampling_values))
sampling_values = [sampling_values]
aaa = PAAAReductor(sampling_values, tf)

# %% slideshow={"slide_type": "fragment"}
aaa_rom = aaa.reduce(tol=1e-4)

# %% slideshow={"slide_type": "fragment"}
aaa_rom

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Bode Plots

# %% slideshow={"slide_type": "-"}
fig, ax = plt.subplots(2, 1, squeeze=False, constrained_layout=True)
_ = tf.bode_plot(w_tf, ax=ax)
_ = tf_rom.transfer_function.bode_plot(w_tf, ax=ax)
_ = aaa_rom.bode_plot(w_tf, ax=ax)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Error System

# %% slideshow={"slide_type": "-"}
aaa_err = tf - aaa_rom

# %% slideshow={"slide_type": "fragment"}
fig, ax = plt.subplots()
_ = tf_err.mag_plot(w_tf, ax=ax, label='TF-IRKA')
_ = aaa_err.mag_plot(w_tf, ax=ax, label='AAA')
_ = ax.legend()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Parametric LTI Models

# %% [markdown] slideshow={"slide_type": "fragment"}
# Cookie model (thermal block) example from [MOR Wiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Thermal_Block).

# %% slideshow={"slide_type": "fragment"}
import scipy.io as spio

mat = spio.loadmat('data/cookie/ABCE.mat')

# %% slideshow={"slide_type": "fragment"}
mat.keys()

# %% slideshow={"slide_type": "fragment"}
A0 = mat['A0']
A1 = 0.2 * mat['A1'] + 0.4 * mat['A2'] + 0.6 * mat['A3'] + 0.8 * mat['A4']
B = mat['B']
C = mat['C']
E = mat['E']

# %% slideshow={"slide_type": "fragment"}
A0

# %% slideshow={"slide_type": "subslide"}
from pymor.operators.numpy import NumpyMatrixOperator

A0op = NumpyMatrixOperator(A0)
A1op = NumpyMatrixOperator(A1)
Bop = NumpyMatrixOperator(B)
Cop = NumpyMatrixOperator(C)
Eop = NumpyMatrixOperator(E)

# %% slideshow={"slide_type": "fragment"}
A0op

# %% slideshow={"slide_type": "subslide"}
from pymor.parameters.functionals import ProjectionParameterFunctional

Aop = A0op + ProjectionParameterFunctional('p') * A1op

# %% slideshow={"slide_type": "fragment"}
Aop

# %% slideshow={"slide_type": "subslide"}
cookie_fom = LTIModel(Aop, Bop, Cop, E=Eop)

# %% slideshow={"slide_type": "fragment"}
cookie_fom

# %% slideshow={"slide_type": "fragment"}
cookie_fom.parameters

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Magnitude Plot

# %% slideshow={"slide_type": "-"}
num_w = 10
num_p = 10
ws = np.logspace(-4, 4, num_w)
ps = np.logspace(-6, 2, num_p)
Hwp = np.empty((num_p, num_w))
for i in range(num_p):
    Hwp[i] = spla.norm(cookie_fom.transfer_function.freq_resp(ws, mu=ps[i]), axis=(1, 2))

# %% slideshow={"slide_type": "fragment"}
from matplotlib.colors import LogNorm

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

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Interpolation

# %% slideshow={"slide_type": "-"}
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

# %% slideshow={"slide_type": "fragment"}
V

# %% slideshow={"slide_type": "subslide"}
from pymor.reductors.basic import LTIPGReductor

pg = LTIPGReductor(cookie_fom, W, V)
cookie_rom = pg.reduce()

# %% slideshow={"slide_type": "fragment"}
cookie_rom

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Error System

# %% slideshow={"slide_type": "-"}
cookie_err = cookie_fom - cookie_rom

# %% slideshow={"slide_type": "fragment"}
Hwp_err = np.empty((num_p, num_w))
for i in range(num_p):
    Hwp_err[i] = spla.norm(cookie_err.transfer_function.freq_resp(ws, mu=ps[i]), axis=(1, 2))

# %% slideshow={"slide_type": "fragment"}
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

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### ROM Poles

# %% slideshow={"slide_type": "-"}
for p in ps:
    poles = cookie_rom.poles(mu=p)
    print(poles.real.max())

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Galerkin Projection

# %% slideshow={"slide_type": "-"}
from pymor.algorithms.pod import pod

VW = V.copy()
VW.append(W)
VW, svals = pod(VW, modes=50)

# %% slideshow={"slide_type": "fragment"}
VW

# %% slideshow={"slide_type": "fragment"}
galerkin = LTIPGReductor(cookie_fom, VW, VW)
cookie_rom_g = galerkin.reduce()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Error System 2

# %% slideshow={"slide_type": "-"}
cookie_err2 = cookie_fom - cookie_rom_g

# %% slideshow={"slide_type": "fragment"}
Hwp_err2 = np.empty((num_p, num_w))
for i in range(num_p):
    Hwp_err2[i] = spla.norm(cookie_err2.transfer_function.freq_resp(ws, mu=ps[i]), axis=(1, 2))

# %% slideshow={"slide_type": "fragment"}
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

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### ROM Poles 2

# %% slideshow={"slide_type": "-"}
for p in ps:
    poles = cookie_rom_g.poles(mu=p)
    print(poles.real.max())

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Nonlinear MOR

# %% [markdown] slideshow={"slide_type": "fragment"}
# Nonlinear RC example from [MOR Wiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Nonlinear_RC_Ladder) (Model 1).
#
# $$
# \begin{align*}
#   \dot{x}(t) & = A x(t) - g(A_0 x(t)) + g(A_1 x(t)) - g(A_2 x(t)) + B u(t) \\
#   y(t) & = C x(t)
# \end{align*}
# $$

# %% slideshow={"slide_type": "fragment"}
import scipy.sparse as sps
from pymor.models.basic import InstationaryModel

# %% slideshow={"slide_type": "fragment"}
# InstationaryModel?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Building an `InstationaryModel`

# %% slideshow={"slide_type": "-"}
n = 1000

A_rc = 41 * sps.diags([(n - 1) * [1], n * [-2], (n - 1) * [1]], [-1, 0, 1], format='csc')

A0_rc = sps.lil_matrix((n, n))
A0_rc[0, 0] = 1
A0_rc = A0_rc.tocsc()

A1_rc = sps.diags([(n - 1) * [1], n * [-1]], [-1, 0], format='lil')
A1_rc[0, 0] = 0
A1_rc = A1_rc.tocsc()

A2_rc = sps.diags([n * [1], (n - 1) * [-1]], [0, 1], format='lil')
A2_rc[-1, -1] = 0
A2_rc = A2_rc.tocsc()

B_rc = np.zeros((n, 1))
B_rc[0, 0] = 1

C_rc = B_rc.T

# %% slideshow={"slide_type": "subslide"}
g = lambda x: np.exp(40 * x) - 40 * x - 1
g_der = lambda x: 40 * np.exp(40 * x) - 40

# %% slideshow={"slide_type": "fragment"}
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace

class ComponentwiseOperator(Operator):
    def __init__(self, mapping, mapping_der=None, dim_source=1, dim_range=1, linear=False,
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

    def jacobian(self, U, mu=None):
        if self.mapping_der is None:
            raise NotImplementedError
        return NumpyMatrixOperator(sps.diags(self.mapping_der(U.to_numpy()[0])),
                                   source_id=self.source_id,
                                   range_id=self.range_id)

class SparseMatrixOperator(NumpyMatrixOperator):
    def restricted(self, dofs):
        matrix_restricted = self.matrix[dofs, :]
        dofs_restricted = matrix_restricted.tocoo().col
        matrix_restricted = matrix_restricted[:, dofs_restricted]
        return SparseMatrixOperator(matrix_restricted), dofs_restricted


# %% slideshow={"slide_type": "subslide"}
g_op = ComponentwiseOperator(g, g_der, dim_source=n, dim_range=n)
A_rc_op = NumpyMatrixOperator(A_rc)
A0_rc_op = SparseMatrixOperator(A0_rc)
A1_rc_op = SparseMatrixOperator(A1_rc)
A2_rc_op = SparseMatrixOperator(A2_rc)
B_rc_op = NumpyMatrixOperator(B_rc)
C_rc_op = NumpyMatrixOperator(C_rc)

# %% slideshow={"slide_type": "subslide"}
from pymor.operators.constructions import LinearInputOperator
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper

# %% slideshow={"slide_type": "fragment"}
T = 2
x0 = A0_rc_op.source.zeros(1)
operator_lin = -A_rc_op
operator_nonlin = g_op @ A0_rc_op - g_op @ A1_rc_op + g_op @ A2_rc_op
operator = operator_lin + operator_nonlin
rhs = LinearInputOperator(B_rc_op)
nt = 500
time_stepper = ExplicitEulerTimeStepper(nt)

rc_fom = InstationaryModel(T, x0, operator, rhs,
                           time_stepper=time_stepper,
                           output_functional=C_rc_op)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Output of the Model

# %% slideshow={"slide_type": "-"}
input_train = 1
rc_output = rc_fom.output(input=input_train)

# %% slideshow={"slide_type": "fragment"}
_ = plt.plot(np.linspace(0, T, nt + 1), rc_output)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### POD

# %% slideshow={"slide_type": "-"}
X_rc = rc_fom.solve(input=input_train)

# %% slideshow={"slide_type": "fragment"}
pod_vec, pod_val = pod(X_rc, rtol=1e-3)

# %% slideshow={"slide_type": "subslide"}
_ = plt.semilogy(range(1, len(pod_val) + 1), pod_val, '.-')

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Galerkin Projection

# %% slideshow={"slide_type": "-"}
from pymor.reductors.basic import InstationaryRBReductor

rb = InstationaryRBReductor(rc_fom, pod_vec)
rc_rom = rb.reduce()

# %% slideshow={"slide_type": "fragment"}
rc_rom

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### POD ROM Output

# %% slideshow={"slide_type": "-"}
rc_rom_output = rc_rom.output(input=input_train)

# %% slideshow={"slide_type": "fragment"}
_ = plt.plot(np.linspace(0, T, nt + 1), rc_output)
_ = plt.plot(np.linspace(0, T, nt + 1), rc_rom_output, '--')

# %% slideshow={"slide_type": "subslide"}
_ = plt.plot(np.linspace(0, T, nt + 1), rc_output - rc_rom_output)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Output with Test Input

# %% slideshow={"slide_type": "-"}
input_test = 'exp(-t)'
rc_output2 = rc_fom.output(input=input_test)
rc_rom_output2 = rc_rom.output(input=input_test)

# %% slideshow={"slide_type": "fragment"}
_ = plt.plot(np.linspace(0, T, nt + 1), rc_output2)
_ = plt.plot(np.linspace(0, T, nt + 1), rc_rom_output2, '--')

# %% slideshow={"slide_type": "subslide"}
_ = plt.plot(np.linspace(0, T, nt + 1), rc_output2 - rc_rom_output2)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### DEIM

# %% slideshow={"slide_type": "-"}
f_X_sol = operator_nonlin.apply(X_rc)

# %% slideshow={"slide_type": "fragment"}
f_X_sol

# %% slideshow={"slide_type": "subslide"}
from pymor.algorithms.ei import deim

interpolation_dofs, collateral_basis, deim_data = deim(f_X_sol)

# %% slideshow={"slide_type": "subslide"}
interpolation_dofs

# %% slideshow={"slide_type": "fragment"}
sorted(interpolation_dofs)

# %% slideshow={"slide_type": "fragment"}
collateral_basis

# %% slideshow={"slide_type": "fragment"}
deim_data

# %% slideshow={"slide_type": "subslide"}
from pymor.operators.ei import EmpiricalInterpolatedOperator

ei_ops = [EmpiricalInterpolatedOperator(op, interpolation_dofs, collateral_basis, True)
          for op in operator_nonlin.operators]

# %% slideshow={"slide_type": "fragment"}
from pymor.operators.constructions import LincombOperator

operator_nonlin_ei = LincombOperator(ei_ops, operator_nonlin.coefficients)

# %% slideshow={"slide_type": "fragment"}
operator_new = operator_lin + operator_nonlin_ei
rc_fom_ei = rc_fom.with_(operator=operator_new)

# %% slideshow={"slide_type": "fragment"}
rc_fom_ei

# %% slideshow={"slide_type": "subslide"}
rb_ei = InstationaryRBReductor(rc_fom_ei, pod_vec)
rc_rom_ei = rb_ei.reduce()

# %% slideshow={"slide_type": "fragment"}
rc_rom_ei

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### POD-DEIM ROM Output

# %% slideshow={"slide_type": "-"}
rc_rom_ei_output2 = rc_rom_ei.output(input=input_test)

# %% slideshow={"slide_type": "fragment"}
_ = plt.plot(np.linspace(0, T, nt + 1), rc_output2)
_ = plt.plot(np.linspace(0, T, nt + 1), rc_rom_ei_output2, '--')

# %% [markdown] slideshow={"slide_type": "slide"}
# # Concluding Remarks

# %% [markdown] slideshow={"slide_type": "subslide"}
# # Further MOR Methods
#
# - second-order systems
# - time-delay systems
# - modal truncation
# - ERA
# - reduced basis methods
# - symplectic methods
# - DMD
# - artificial neural networks

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Using pyMOR
#
# - installation: https://github.com/pymor/pymor#readme
# - documentation: https://docs.pymor.org
# - GitHub issues: https://github.com/pymor/pymor/issues
# - GitHub discussions: https://github.com/pymor/pymor/discussions
# - pyMOR community meetings: https://github.com/pymor/pymor/discussions/categories/announcements
# - pyMOR School: https://school.pymor.org

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Contributing to pyMOR
#
# - developer documentation: https://docs.pymor.org/latest/developer_docs.html
# - get attribution via `AUTHORS.md`
# - become contributor with push access to feature branches
# - become main developer with full control over the project
