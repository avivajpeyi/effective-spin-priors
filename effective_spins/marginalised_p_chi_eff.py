# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: PyCharm (bilby-master)
#     language: python
#     name: pycharm-ab23381c
# ---

# # Imports and configs

# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import  matplotlib
from matplotlib import rc
from scipy.signal import fftconvolve
from bilby.core.prior import Cosine, Interped, Uniform, Constraint
from bilby.core.prior import PriorDict
import bilby
from priors import (
    chi_effective_prior_from_isotropic_spins,
    get_marginalised_chi_eff, 
    calculate_xeff, 
    calculate_xp,
    calculate_xp_given_xeff
)
from agn_utils.plotting import overlaid_corner

# %config InlineBackend.print_figure_kwargs = {'facecolor': "w"}
# %matplotlib inline
rc("text", usetex=True)
plt.style.reload_library()
plt.style.use(['science', 'grid', 'notebook'])
plt.rcParams['font.size'] = 20


# # Isotropic chi_effective priors

# + pycharm={"name": "#%%\n"}
def get_traditional_prior():
    priors = PriorDict()
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a1'] = Uniform(minimum=0, maximum=1)
    priors['a2'] = Uniform(minimum=0, maximum=1)
    priors['cos2'] = Uniform(minimum=-1, maximum=1)
    priors['cos1'] = Uniform(minimum=-1, maximum=1)
    return priors
    
def get_traditional_samples(num_samples=10 ** 5):
    priors = get_traditional_prior()
    s= pd.DataFrame(priors.sample(num_samples))
    s['sin1'] = np.sqrt(1-s.cos1**2)
    s['sin2'] = np.sqrt(1-s.cos2**2) 
    s['tan1'] = s['sin1'] / s['cos1']
    s['tan2'] = s['sin2'] / s['cos2']
    s['xeff'] = calculate_xeff(s.a1, s.a2, s.cos1, s.cos2, s.q)
    xp = calculate_xp(s.a1, s.a2, s.q, s.sin1, s.sin2)
    xp_given_xeff = calculate_xp_given_xeff(
        s.xeff, s.a1, s.a2, s.q, s.cos1, s.cos2, s.tan1, s.tan2
    )
    return s

    
def plot_funct_and_samples(func, samples, limits, labels, func_kwargs={}, bins=50):
    xvals = np.linspace(limits[0], limits[1], 300)
    yvals = func(xvals, **func_kwargs)
    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(111)
    ax1.hist(samples, density=True, bins=bins)
    ax1.plot(xvals, yvals, color="black")
    ax1.xaxis.grid(True, which="major", ls=":", color="grey")
    ax1.yaxis.grid(True, which="major", ls=":", color="grey")
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel(labels[0], fontsize=18)
    ax1.set_ylabel(labels[1], fontsize=18)
    ax1.set_xlim(limits[0], limits[1])
    return fig

def plot_xeff():
    s = get_traditional_samples()
    fig = plot_funct_and_samples(
        get_marginalised_chi_eff,
        s.xeff,
        [-1, 1],
        [r"$\chi_{\rm eff}$", r"$p(\chi_{\rm eff})$"],
    )
    
plot_xeff()


# -

# # Finding Limits of Integration
# \begin{align}
# \newcommand{\xp}{{\chi_{\text{p}}}}
# \newcommand{\xeff}{{\chi_{\text{eff}}}}
#     \xeff &= \frac{\chi_1\cos\theta_1 + q\chi_2\cos\theta_2}{1+q} \\
#     \xp &= \rm{Max}\left[ \chi_1\sin\theta_1,\ \chi_2\sin\theta_2\ q\frac{4q+3}{4+3q} \right]
# \end{align}
# ## Limits on $\xeff$ 
# The max/min $\cos x = -1,1$, hence, the limits on $\xeff$ are:
# \begin{equation}
# -\frac{\chi_1 + q\chi_2}{1+q}  \leq \xeff \leq \frac{\chi_1 + q\chi_2}{1+q}
# \end{equation}
#
# ## Limits on $\cos\theta_2$
# Taking $X_1=\chi_1/(1+q)$ and $X_2=q\chi_2/(1+q)$ Rearranging the equation for $\chi_{\text{eff}}$, we can obtain
# \begin{align}
# \newcommand{\xeff}{{\chi_{\text{eff}}}}
# &X_1\cos\theta_1 = \xeff - X_2\cos\theta_2\\
# \implies &-X_1 \leq \xeff - X_2\cos\theta_2 \leq X_1\\
# \implies &\frac{-X_1-\xeff}{-X_2} \geq  \cos\theta_2 \geq \frac{X_1-\xeff}{-X_2}\\
# \implies &\frac{X_1+\xeff}{X_2} \geq  \cos\theta_2 \geq \frac{-X_1+\xeff}{X_2}\\
# \implies &\frac{-\chi_1+(q+1)\xeff}{q\chi_2}\leq \cos\theta_2\leq \frac{\chi_1+(q+1)\xeff}{q\chi_2}
# \end{align}
# which sets limits on the values for $\cos\theta_2$ that can be drawn.
#
# Additionally, we can calculate $\cos\theta_2$ from 
# \begin{equation}
# \newcommand{\xeff}{{\chi_{\text{eff}}}}
# \cos\theta_2 = \frac{-\chi_1\cos(\theta_1)+(1+q)\xeff}{\chi_2q}
# \end{equation}

# ## Implementing Limits

# +
def calc_cos2(q, a1, a2, xeff, cos1):
    return (xeff*(1+q)-a1*cos1)/(a2*q)

def calc_cos1(q, a1, a2, xeff, cos2):
    return (xeff*(1+q)-q*a2*cos2)/(a1)

@np.vectorize
def calc_xeff_limit(a1,a2,q):
    lim = (a1+q*a2) / (q+1)
    return lim

@np.vectorize
def calc_cos2_limits(q, a1, a2, xeff):
    lim1 = (+a1+(q+1)*xeff)/(q*a2)
    lim2 = (-a1+(q+1)*xeff)/(q*a2)
    lower_lim = np.minimum(lim1, lim2)
    upper_lim = np.maximum(lim1, lim2)
    return np.maximum(lower_lim, -1), np.minimum(upper_lim, 1)

@np.vectorize
def cos2_lower_lim(q, a1, a2, xeff):
    return min(calc_cos2_limits(q, a1, a2, xeff))

@np.vectorize
def cos2_upper_lim(q, a1, a2, xeff):
    return max(calc_cos2_limits(q, a1, a2, xeff))

# -
# ### plotting cos1 cos2


# +
def plot_cos_given_other_params(q, a1,a2,xeff):
    plt.close('all')
    cos_lims = calc_cos2_limits(q, a1, a2, xeff)
    cos_2_vals = np.linspace(-1, 1, num=300)
    cos_1_vals = calc_cos1(q, a1, a2, xeff, cos_2_vals)
    fig, ax = plt.subplots()
    xeff_lim = calc_xeff_limit(a1, a2, q)
    valid_cos2 = [cos_lims[0] <= cos_2_vals[i] <= cos_lims[1] for i in range(len(cos_2_vals))]
    colors = ["green" if v else "red" for v in valid_cos2]
    ax.plot(cos_2_vals, cos_1_vals, c='k')
    ylims = min(cos_1_vals), max(cos_1_vals)
    ax.axhspan(min(ylims), -1,  alpha=0.5, color='orange')
    ax.axhspan(1, max(ylims),  alpha=0.5, color='orange')
    ax.axvspan(-1, cos_lims[0],  alpha=0.5, color='red')
    ax.axvspan(cos_lims[1], 1, alpha=0.5, color='red')
    ax.set_ylabel(r"$\cos\theta1$")
    ax.set_xlabel(r"$\cos\theta2$")
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    xeff_valid = -xeff_lim <= xeff <= xeff_lim 
    xeff_str = f"$\chi_e={xeff:.1f}$" if xeff_valid else "$\chi_e=$INVALID"
    label= (f"$q={q:.1f}$" + "\n"
            f"$a_1={a1:.1f}$"+ "\n" 
            f"$a_2={a2:.1f}$"+"\n"
            +xeff_str+"\n"
            f"$lim=[{cos_lims[0]:.1f},{cos_lims[1]:.1f}]%$"
           )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    
    
plot_cos_given_other_params(q=0.9, a1=0.2, a2=0.9, xeff=-0.1)
# -




plot_interactive = True
if plot_interactive:
    from ipywidgets import interact, FloatSlider
    import ipywidgets as widgets
    interact(
        plot_cos_given_other_params,
        q=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False), 
        a1=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
        a2=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
        xeff=FloatSlider(min=-1, max=1, step=0.1, continuous_update=False)
    )


# ## Sampling methods for limits on priors

# +
def get_sample_in_lim(lim, pri):
    """Sample in limited range for prior"""
    og_min, og_max = pri.minimum, pri.maximum
    pri.minimum,pri.maximum = lim[0], lim[1]
    s = pri.sample(1)
    pri.minimum, pri.maximum = og_min, og_max
    return s

def get_sample_in_lim_with_repeated_sampling(lim, pri, threshold=0.001):
    """Sample in full prior until sample obtained in limited range"""
    p_for_samp_in_lim = 0.0
    for i in np.linspace(lim[0], lim[1], n=100):
        p_for_samp_in_lim += pri.prob(i)
    s = np.nan
    if p_for_samp_in_lim >= threshold:
        s = pri.sample(1)
        while not (lim[0]<=s<=lim[1]):
            s = pri.sample(1)
    return s



# -

# ## Method to obtain correct prior: rejection sampling
#
# 1. Sample q, a1, a2, xeff, cos2
# 2. reject invalid xeff
# 3. reject invalid cos2
# 4. Calc cos1 and check xeff(q,a1,a2,cos1,cos2) = xeff

# +
def check_samples(s):
    assert len(s) == len(filter_invalid_cos2(s))
    assert len(s) == len(filter_invalid_xeff(s))

def filter_invalid_cos2(s):
    lower = cos2_lower_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    upper = cos2_upper_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    s['valid_cos2'] = (
        (lower <= s['cos_tilt_2']) & (s['cos_tilt_2'] <= upper)
    )
    print(f"# valid cos: {sum(s['valid_cos2'])}/{len(s['valid_cos2'])}")
    s = s[s['valid_cos2'] == True]
    s.pop("valid_cos2")
    return s

def filter_invalid_xeff(s):
    s['valid_xeff'] = (
        (s['a_1']+s['q']*s['a_2']) / (s['q']+1) >= np.abs(s['xeff'])
    )
    print(f"# valid xeff: {sum(s['valid_xeff'])}/{len(s['valid_xeff'])}")
    s = s[s['valid_xeff'] == True]
    return s

def filter_cos_out_of_bounds(s):
    for i in [1,2]:
        s[f'valid_c{i}'] = (-1 <= s[f'cos_tilt_{i}'])& (s[f'cos_tilt_{i}']<= 1)
    s['valid_c'] = s['valid_c1'] & s['valid_c2']
    print(f"# valid cos: {sum(s['valid_c'])}/{len(s['valid_c'])}")
    s = s[s['valid_c'] == True]
    s.drop(["valid_c", "valid_c1", "valid_c2"],axis=1)
    return s
    
@np.vectorize
def get_valid_cos2_samples(a1,a2, q, xeff, cos_prior):
    """-1 <= (xeff(q+1)-a1)/(qa2) <= cos2 <= (xeff(q+1)+a1)/(qa2) <= 1"""
    lower = cos2_lower_lim(q, a1, a2, xeff)
    upper = cos2_upper_lim(q, a1, a2, xeff)
    return get_sample_in_lim([lower, upper], cos_prior)

@np.vectorize
def get_valid_xeff_samples(q, a1, a2):
    xeffs = np.linspace(-1, 1, 300)
    p_xeff = chi_effective_prior_from_isotropic_spins(q=q, aMax=np.maximum(a1, a2), xs=xeffs)
    xeff_prior = Interped(xeffs, p_xeff, minimum=-1, maximum=1)
    xeff_lim = calc_xeff_limit(a1=a1, a2=a2, q=q)
    s = get_sample_in_lim([-xeff_lim, xeff_lim], xeff_prior)
    assert -xeff_lim <= s <= xeff_lim
    return s

def get_samples(num=10):
    priors = PriorDict()
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a_1'] = Uniform(minimum=0, maximum=1)
    priors['a_2'] = Uniform(minimum=0, maximum=1)
    s = pd.DataFrame(priors.sample(num))
    s['xeff'] = get_valid_xeff_samples(q=s.q, a1=s.a_1, a2=s.a_2)
    cos_prior = Uniform(minimum=-1, maximum=1)
    s['cos_tilt_2'] = get_valid_cos2_samples(a1=s.a_1, a2=s.a_2, q=s.q,  xeff=s.xeff, cos_prior=cos_prior)
    s['cos_tilt_1'] = calc_cos1(q=s.q, a1=s.a_1, a2=s.a_2, xeff=s.xeff, cos2=s.cos_tilt_2)
    s['xeff_valid'] = np.abs(s['xeff']) <= (s['a_1']+s['q']*s['a_2']) / (s['q']+1)
    s['lower_lims'] =  cos2_lower_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    s['upper_lims'] = cos2_upper_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    check_cos_samples(s)
    return s

def get_samples_without_coslimits(num=10):
    priors = PriorDict()
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a_1'] = Uniform(minimum=0, maximum=1)
    priors['a_2'] = Uniform(minimum=0, maximum=1)
    priors['cos_tilt_2'] = Uniform(minimum=-1, maximum=1)
    s = pd.DataFrame(priors.sample(num))
    s['xeff'] = get_valid_xeff_samples(q=s.q, a1=s.a_1, a2=s.a_2)
    s['cos_tilt_1'] = calc_cos1(q=s.q, a1=s.a_1, a2=s.a_2, xeff=s.xeff, cos2=s.cos_tilt_2)
    return s

# Simple test for sample getters 
assert len(get_samples()) != 0
assert len(get_samples_without_coslimits()) != 0


# -


# ### Plots of Samples

# +
def get_prior_with_xeff():
    """This is currently incorrect (xeff + cos2 limits not quite correct!)"""
    priors = PriorDict()
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a_1'] = Uniform(minimum=0, maximum=1)
    priors['a_2'] = Uniform(minimum=0, maximum=1)
    priors['cos_tilt_2'] = Uniform(minimum=-1, maximum=1)
    xeffs = np.linspace(-1, 1, 300)
    p_xeff = get_marginalised_chi_eff(xeffs) 
    priors['xeff'] = Interped(xeffs, p_xeff, minimum=-1, maximum=1)
    priors['cos_tilt_1'] =  Uniform(minimum=-1, maximum=1)
    return priors

def plot_samples(s, param, sample_ranges=None):
    p = list(s.columns.values)
    r = bilby.core.result.Result(search_parameter_keys=p, parameter_labels=p, posterior=s)
    fig = r.plot_corner(parameters=param, priors=get_prior_with_xeff(), range=sample_ranges)
    return fig 


# -

# %%time
s = get_samples(10000)
plot_samples(s, ["a_1", "a_2", 'xeff','cos_tilt_2', 'cos_tilt_1'])

# %%time
s_wo = get_samples_without_coslimits(10000)
plot_samples(s, ['a_1', 'a_2', 'xeff', 'cos_tilt_2', 'cos_tilt_1'])

# %%time
filtered_s = filter_invalid_xeff(s)
filtered_s = filter_invalid_cos2(filtered_s)
plot_samples(s, ['a_1', 'a_2', 'xeff', 'cos_tilt_2', 'cos_tilt_1'])

# # Calculat p(xeff| q, a1, a2)

# \begin{align}
# \newcommand{\xp}{{\chi_{\text{p}}}}
# \newcommand{\xeff}{{\chi_{\text{eff}}}}
#     \xeff &= \frac{\chi_1\cos\theta_1 + q\chi_2\cos\theta_2}{1+q}
# \end{align}
#
# Now assuming we are given $\{\chi_1, \chi_2, q\}$, we can write
#
# \begin{align}
#     \pi(\xeff| \chi_1, \chi_2, q) &= \pi\left(\frac{\chi_1}{1+q}\ \pi(\cos \theta_1) \right) \odot \pi\left(\frac{q\chi_2}{1+q}\ \pi(\cos \theta_2) \right)
# \end{align}
#
# Taking $X=\cos \theta_1=U[a,b]=(1/(b-a))\mathcal{1}_{a\leq x\leq b}$ and $Y=\cos \theta_2=U[c,d]=(1/(d-c))\mathcal{1}_{c\leq x\leq d}$, we can calculate $Z=X+Y$ with
#
# \begin{align}
# P_Z(z) &= \int_{\infty}^{\infty} dx\ P_X(x) P_Y(z-x) \\
#        &= (1/(b-a)(c-d)) \int_{\infty }^{\infty} dx\ \mathcal{1}_{a\leq x\leq b}  \mathcal{1}_{c\leq z-x\leq d} \\
#        &= (1/(b-a)(c-d)) \int_{\infty }^{\infty} dx\ \mathcal{1}_{a\leq x\leq b} \mathcal{1}_{z-d\leq x\leq z-c} \\
#        &= (1/(b-a)(c-d)) \int_{\infty }^{\infty} dx\ \mathcal{1}_{\rm{max}(a, z-d) \leq x\leq \rm{min}(b, z-c)} \\
#        &= (1/(b-a)(c-d))  \rm{min}(b, z-c) - \rm{max}(a, z-d) \\
# \end{align}
#
# Now, for the case where $X=U[]$, $Y=[]$
#
# See [stackExchange](https://math.stackexchange.com/questions/1486611/convolution-of-two-uniform-random-variables)

# +
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import signal


@np.vectorize
def rory_p_xeff_given_a1_a2_q(chi_eff, a1, a2, q):
    chi_eff_min = -a1/(1+q) - a2*q/(1+q)
    chi_eff_max = a1/(1+q)  + a2*q/(1+q)
    if (chi_eff < chi_eff_min  ) or (chi_eff_max < chi_eff):
        return 0
    else:
        return (1+q)**2 / (4*a1*a2*q) * ( np.minimum(a1/(1+q), chi_eff + a2*q/(1+q)) -
                               np.maximum(-a1/(1+q), chi_eff - a2*q/(1+q)) ) 

@np.vectorize
def p_xeff_given_a1_a2_q(xeff, a1, a2, q):
    xeff_lim = a1/(1+q)  + a2*q/(1+q)
    c1 = [-a1 / (q+1), a1 / (q+1)]
    c2 = [-a2*q / (q+1), a2*q / (q+1)]
    constant = 1/((c1[1]-c1[0])*(c2[1]-c2[0]))
    min_term = np.minimum(c1[1], xeff-c2[0])
    max_term = np.maximum(c1[0], xeff-c2[1])
    p = constant * (min_term-max_term)
    if (xeff < -xeff_lim  ) or (xeff_lim < xeff):
        p = 0
    return p

def plot_xeff_given_a1_a2_q(a1,a2,q):
    xlims = calc_xeff_limit(a1,a2,q)
    xs = np.arange(-xlims,xlims, 0.001)
    pxs = p_xeff_given_a1_a2_q(xs, a1, a2, q)
    rpxs = rory_p_xeff_given_a1_a2_q(xs, a1, a2, q)
    unip = Uniform(minimum=-1, maximum=1)
    samps = calculate_xeff(a1=a1, a2=a2, cos1=unip.sample(10000), cos2=unip.sample(10000), q=q)
    fig, ax = plt.subplots()
    ax.plot(xs, pxs, "o-", linewidth=3, label="Avi")
    ax.plot(xs, rpxs, "o--", linewidth=3, label="Rory")
    _ = ax.hist(samps, density=True, bins=50, label="samples")
    ax.set_xlim(-1,1)
    label= (f"$q={q:.1f}$" + "\n"
            f"$a_1={a1:.1f}$"+ "\n" 
            f"$a_2={a2:.1f}$"+"\n"
            f"$lim=[{-xlims:.1f},{xlims:.1f}]%$"
           )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
    ax.set_ylabel("p(xeff|a1,a2,q)")
    ax.set_xlabel("xeff")
    plt.legend(loc='upper right')
    plt.tight_layout()

    
interact(
    plot_xeff_given_a1_a2_q,
    q=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False), 
    a1=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
    a2=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
)

# +
@np.vectorize
def get_valid_xeff_samples(q, a1, a2):
    xeffs = np.linspace(-1, 1, 300)
    p_xeff = p_xeff_given_a1_a2_q(xeff=xeffs, a1=a1, a2=a2, q=q)
    xeff_lim = calc_xeff_limit(a1=a1, a2=a2, q=q)
    xeff_prior = Interped(xeffs, p_xeff, minimum=-xeff_lim, maximum=xeff_lim)
    s = get_sample_in_lim([-xeff_lim, xeff_lim], xeff_prior)
    assert -xeff_lim <= s <= xeff_lim
    return s
    

def get_samples(num=10):
    priors = PriorDict()
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a_1'] = Uniform(minimum=0, maximum=1)
    priors['a_2'] = Uniform(minimum=0, maximum=1)
    s = pd.DataFrame(priors.sample(num))
    s['xeff'] = get_valid_xeff_samples(q=s.q, a1=s.a_1, a2=s.a_2)
    cos_prior = Uniform(minimum=-1, maximum=1)
    s['cos_tilt_2'] = get_valid_cos2_samples(a1=s.a_1, a2=s.a_2, q=s.q,  xeff=s.xeff, cos_prior=cos_prior)
    s['cos_tilt_1'] = calc_cos1(q=s.q, a1=s.a_1, a2=s.a_2, xeff=s.xeff, cos2=s.cos_tilt_2)
    s['xeff_valid'] = np.abs(s['xeff']) <= (s['a_1']+s['q']*s['a_2']) / (s['q']+1)
    s['lower_lims'] =  cos2_lower_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    s['upper_lims'] = cos2_upper_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    check_cos_samples(s)
    return s


# -


s = get_samples(10000)

# +
from tqdm.auto import tqdm

# def inverse_sample_function(dist, pnts, a1, a2, q, x_min=chi_eff_min, x_max=chi_eff_max, n=1e5):
#     x = np.linspace(x_min, x_max, int(n))
#     cumulative = np.cumsum(dist(x, a1, a2, q))
#     cumulative -= cumulative.min()
#     f = interp1d(cumulative/cumulative.max(), x)
#     return f(np.random.random(pnts))

# def get_marginalised_p_xeff(xs):
#     priors = PriorDict()
#     priors['q'] = Uniform(minimum=0, maximum=1)
#     priors['a1'] = Uniform(minimum=0, maximum=1)
#     priors['a2'] = Uniform(minimum=0, maximum=1)
#     v = np.linspace(0,1, 301)[1:] # skip over q,a1,a2=0
#     p_xeff = np.zeros(len(v))
#     for q in tqdm(v):
#         for a1 in tqdm(v):
#             for a2 in tqdm(v):
#                 others = dict(a1=a1, a2=a2, q=q)
#                 p_xeff_given_other=p_xeff_given_a1_a2_q(xeff=xs, **others)
#                 p_other = priors.prob(others)
#                 p_xeff += np.array(p_xeff_given_other * p_other)
#     p_xeff = p_xeff / len(v)**3
#     return p_xeff

# def get_marginalised_p_xeff(xs):
#     priors = PriorDict()
#     priors['q'] = Uniform(minimum=0, maximum=1)
#     priors['a1'] = Uniform(minimum=0, maximum=1)
#     priors['a2'] = Uniform(minimum=0, maximum=1)
#     samples = pd.DataFrame(priors.sample(1000)).to_dict('records')
#     p_xeff = np.zeros(len(xs))
#     for s in tqdm(samples, total=len(samples)):
#         p_xeff_given_other=p_xeff_given_a1_a2_q(xeff=xs, **s)
#         p_xeff += np.array(p_xeff_given_other)
#     p_xeff = p_xeff /len(samples)**3
#     return p_xeff


# def get_prior_with_xeff():
#     """This is currently incorrect (xeff + cos2 limits not quite correct!)"""
#     priors = PriorDict()
#     priors['q'] = Uniform(minimum=0, maximum=1)
#     priors['a_1'] = Uniform(minimum=0, maximum=1)
#     priors['a_2'] = Uniform(minimum=0, maximum=1)
#     priors['cos_tilt_2'] = Uniform(minimum=-1, maximum=1)
#     xeffs = np.linspace(-1, 1, 300)
#     p_xeff = get_marginalised_chi_eff(xeffs) 
#     priors['xeff'] = Interped(xeffs, p_xeff, minimum=-1, maximum=1)
#     priors['cos_tilt_1'] =  Uniform(minimum=-1, maximum=1)
#     return priors

# def plot_samples(s, param, sample_ranges=None, **kwargs):
#     p = list(s.columns.values)
#     r = bilby.core.result.Result(search_parameter_keys=p, parameter_labels=p, posterior=s)
#     fig = r.plot_corner(parameters=param, priors=get_prior_with_xeff(), range=sample_ranges, **kwargs)
#     return fig 



plot_samples(s, ["a_1", "a_2", 'xeff','cos_tilt_2', 'cos_tilt_1'])
# get_prior_with_xeff()
# -

 np.linspace(0,1, 3)[1:]

trad = get_traditional_samples()
trad['a_1'] = trad['a1']
trad['a_2'] = trad['a2']
trad['cos_tilt_1'] = trad['cos1']
trad['cos_tilt_2'] = trad['cos2']
plot_samples(trad, ["a_1", "a_2", 'xeff','cos_tilt_2', 'cos_tilt_1'],suptitle="BLAH")



# +
def plot_margin_xeff():
    s = get_traditional_samples()
    fig = plot_funct_and_samples(
        get_marginalised_p_xeff,
        s.xeff,
        [-1, 1],
        [r"$\chi_{\rm eff}$", r"$p(\chi_{\rm eff})$"],
    )
    
plot_xeff()
plot_margin_xeff()
# -

s = get_traditional_samples(1e8)

s.describe().T

s[(s['xeff']>0.7) & (s['xeff']<0.71)].describe()

s[(s['xeff']>0.7) & (s['xeff']<0.71)]
