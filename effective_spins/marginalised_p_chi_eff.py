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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import rc
from scipy.signal import fftconvolve
from bilby.core.prior import Cosine, Interped, Uniform, Constraint
from bilby.core.prior import PriorDict
import bilby
from priors import (
    chi_effective_prior_from_isotropic_spins,
    chi_p_prior_from_isotropic_spins,
)
from agn_utils.plotting import overlaid_corner




# %config InlineBackend.print_figure_kwargs = {'facecolor': "w"}
# %matplotlib inline
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
plt.style.reload_library()
plt.style.use(['science', 'grid', 'notebook'])
plt.rcParams['font.size'] = 20


# # Isotropic chi_effective priors

# + pycharm={"name": "#%%\n"}
def q_factor(q):
    return ((3.0 + 4.0 * q) / (4.0 + 3.0 * q)) * q


def calculate_xp_given_xeff_from_samples(xeff, a1, a2, q, cos1, cos2, tan1, tan2):
    case1 = (xeff * (1 + q) - a2 * q * cos2) * tan1
    case2 = (xeff + q * xeff - a1 * cos1) * tan2 * q_factor(q)
    return np.maximum(case1, case2)


def calculate_xp_from_samples(a1, a2, q, sin1, sin2):
    case1 = a1 * sin1
    case2 = a2 * sin2 * q_factor(q)
    return np.maximum(case1, case2)


def calculate_xeff(a1, a2, cos1, cos2, q):
    return ((a1 * cos1) + (q * a2 * cos2)) / (1.0 + q)


def get_traditional_samples(num_samples=10 ** 5):
    a1s = scipy.stats.uniform(0, 1).rvs(num_samples)
    a2s = scipy.stats.uniform(0, 1).rvs(num_samples)
    cos1s = scipy.stats.uniform(-1, 2).rvs(num_samples)
    cos2s = scipy.stats.uniform(-1, 2).rvs(num_samples)
    qs = scipy.stats.uniform(0, 1).rvs(num_samples)
    sin1s = np.sqrt(1.0 - cos1s ** 2)
    sin2s = np.sqrt(1.0 - cos2s ** 2)
    tan1s = sin1s / cos1s
    tan2s = sin2s / cos2s
    xeff = calculate_xeff(a1s, a2s, cos1s, cos2s, qs)
    xp = calculate_xp_from_samples(a1s, a2s, qs, sin1s, sin2s)
    xp_given_xeff = calculate_xp_given_xeff_from_samples(
        xeff, a1s, a2s, qs, cos1s, cos2s, tan1s, tan2s
    )

    return pd.DataFrame(
        dict(
            a1s=a1s,
            a2s=a2s,
            cos1s=cos1s,
            cos2s=cos2s,
            qs=qs,
            sin1s=sin1s,
            sin2s=sin2s,
            tan1s=tan1s,
            tan2s=tan2s,
            theta1s=np.arcsin(sin1s),
            theta2s=np.arcsin(sin2s),
            xp=xp,
            xp_given_xeff=xp_given_xeff,
            xeff=xeff,
        )
    )


def get_marginalised_chi_p(xs):
    qs = np.linspace(0, 1, 300)
    p_xp = np.zeros(len(qs))
    for q in qs[1:]:  # skip over q=0
        p_xp += np.array(chi_p_prior_from_isotropic_spins(q, aMax=1, xs=xs))
    p_xp = p_xp / (len(qs) - 1)
    return p_xp


def get_marginalised_chi_eff(xs):
    qs = np.linspace(0, 1, 300)
    p_xeff = np.zeros(len(qs))
    for q in qs[1:]:  # skip over q=0
        p_xeff += np.array(chi_effective_prior_from_isotropic_spins(q, aMax=1, xs=xs))
    p_xeff = p_xeff / (len(qs) - 1)
    return p_xeff


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


def plot_xp():
    s = get_traditional_samples()

    fig = plot_funct_and_samples(
        get_marginalised_chi_p, s.xp, [0, 1], [
            r"$\chi_{\rm p}$", r"$p(\chi_{\rm p})$"]
    )


def product_distribution(x, y, bins):
    pdf_x, x_bins = np.histogram(x, bins, density=True)
    pdf_y, y_bins = np.histogram(y, bins=x_bins, density=True)
    xi = x_bins[1:]
    yi = y_bins[1:]
    dx = x_bins[1] - x_bins[0]
    pdf_z = pdf_x * pdf_y * dx * (1.0 / np.abs(xi))
    zi = xi * yi
    return pdf_x * pdf_y, zi


def sum_distribution(x, y, bins):
    pdf_x, x_bins = np.histogram(x, bins, density=True)
    pdf_y, y_bins = np.histogram(y, bins=x_bins, density=True)
    xi = x_bins[1:]
    yi = y_bins[1:]
    pdf_z = fftconvolve(pdf_x, pdf_y, "same")
    zi = xi + yi
    return pdf_z, zi


def case1_xp_given_xeff():
    """
        xp = tan1 (xeff(1+q) - qa2cos2)
    P(xp|xeff)
        = P(tan1 xeff (1+q) - tan1 q a2 cos2)
        = Convolve[ P(tan1 xeff (1+q)), P(-tan1 q a2 cos2)]
        = Convolve[ xeff Prod[P(tan1), P(1+q)], Prod[P(-tan1), P(q a2 cos2)]]
    """
    pass


def pdf_qchicos(xs):
    return (1 / 4) * np.log(np.abs(xs)) ** 2


def pdf_tan(xs):
    return np.abs(xs / (1 + xs ** 2) ** 1.5) / 2


@np.vectorize
def a1_given_xeff(xeff, q, a2, theta1, theta2):
    a1 = np.abs((xeff * (1 + q) - a2 * q *
                np.cos(theta2) * (1 / np.cos(theta1))))
    return a1


@np.vectorize
def a2_given_xeff(xeff, q, a1, theta1, theta2):
    a2 = np.abs(a1_given_xeff(xeff, q, a1, theta2, theta1)/q)
    return a2


def get_linear_spaced_bins(num):
    bins = np.arange(-10, 10, (20/num))
    bins = (bins[1:] + bins[:-1]) / 2
    return bins


# -

plot_xeff()
plot_xp()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Calculating p(chi_p|chi_eff)
# -

s = get_traditional_samples()
plot_funct_and_samples(pdf_qchicos, s.a1s * s.cos1s *
                       s.qs, [-1, 1], [r"$q a \cos$", "$p(q a \cos)$"])
plot_funct_and_samples(
    pdf_tan, s.tan1s, [-10, 10], [r"$tan$", "$p(tan)$"], bins=get_linear_spaced_bins(100))


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
# \implies &-\frac{-X_1-\xeff}{X_2} \leq  \cos\theta_2 \leq \frac{X_1-\xeff}{X_2}\\
# \implies &\frac{-\chi_1+(q+1)\xeff}{q\chi_2}\leq \cos\theta_2\leq \frac{\chi_1+(q+1)\xeff}{q\chi_2}
# \end{align}
# which sets limits on the values for $\cos\theta_2$ that can be drawn. 
#
# Additionally, we can calculate $\cos\theta_2$ from 
# \begin{equation}
# \newcommand{\xeff}{{\chi_{\text{eff}}}}
# \cos\theta_2 = \frac{-\chi_1\cos(\theta_1)+(1+q)\xeff}{\chi_2q}
# \end{equation}

# +
@np.vectorize
def calc_cos2_limits(q, a1, a2, xeff):
    lim1 = (+a1+(q+1)*xeff)/(q*a2)
    lim2 = (-a1+(q+1)*xeff)/(q*a2)
    lower_lim = np.minimum(lim1, lim2)
    upper_lim = np.maximum(lim1, lim2)
    return np.maximum(lower_lim, -1), np.minimum(upper_lim, 1)



def calc_cos2(q, a1, a2, xeff, cos1):
    return (xeff*(1+q)-a1*cos1)/(a2*q)


def calc_cos1(q, a1, a2, xeff, cos2):
    return (xeff*(1+q)-q*a2*cos2)/(a1)


def filter_dataframe(dataframe: pd.DataFrame, **kwargs):
    """Filter dataframe based on min-max kwargs.

    :param dataframe: the dataframe being filtered
    :param kwargs: the key of the dataframe followed by '_min' or '_max' and the value
    that the key should be set to. For example,
    """
    before_len = len(dataframe)
    if kwargs:
        strcond = "&".join((make_cond(k, v) for k, v in kwargs.items()))
        dataframe = dataframe.query(strcond)
    after_len = len(dataframe)
    print(
        f"Filtered with the condition: {strcond}. {before_len} samples --> {after_len} samples"
    )
    return dataframe


def make_cond(k, v):
    """Create a string format of a key and value condition.

    :param k:
    :param v:
    :return:
    """
    if len(k) < 5:
        raise (ValueError("Arg too short {}".format(k)))
    if k.endswith("_min"):
        return "({}>={})".format(k[:-4], v)
    if k.endswith("_max"):
        return "({}<={})".format(k[:-4], v)

def downsample_case1(s):
    c = r"$xeff>0, \cos1>0$"
    return filter_dataframe(s, xeff_min=0, cos_tilt_1_min=0), c


def downsample_case2(s):
    c = r"$xeff>0, \cos1<0$"
    return filter_dataframe(s, xeff_min=0, cos_tilt_1_max=0), c


def downsample_case3(s):
    c = r"$xeff<0, \cos1>0$"
    return filter_dataframe(s, xeff_max=0, cos_tilt_1_min=0), c


def downsample_case4(s):
    c = r"$xeff<0, \cos1<0$"
    return filter_dataframe(s, xeff_max=0, cos_tilt_1_max=0), c


def plot_samples(case_func):
    p = ['xeff', 'cos_tilt_1', 'cos_tilt_2']
    s = get_samples()
    case, l = case_func(s)
    overlaid_corner(
        samples_list=[s, case],
        sample_labels=["All", l],
        params=p,
        samples_colors=['orange', 'olive'], quants=False,
    )
    print(case[p].describe().T[["min", "max"]])
# -
# [](test.png)


# ## Method 1: Prior-resampling
# 1. Sample q, a1, a2, xeff, cos2
# 2. If invalid sample, resample
# 3. Calc cos1 and check xeff(q,a1,a2,cos1,cos2) = xeff


# +

def convert_a1_a2_xeff_to_z(parameters):
    """
    Function to convert between sampled parameters and constraint parameter.
    Constraint1: xeff_constraint = a1+a2 >= abs(xeff) 
    Constraint2: cos2_constraint = abs(limit) >= abs(cos2)

    Returns
    -------
    dict: Dictionary with constraint parameters
    """
    converted_parameters = parameters.copy()
    converted_parameters['xeff_constraint'] = (parameters['a_1'] + \
        parameters['a_2']*parameters['q']/(1+parameters['q'])) - np.abs(parameters['xeff'])

#     cos_lim = calc_cos2_limits(
#         q=parameters['q'], a1=parameters['a_1'], a2=parameters['a_2'], xeff=parameters['xeff'])
#     converted_parameters['cos_constraint'] = - \
#         np.abs(parameters['cos_tilt_2']) + cos_lim

#     converted_parameters['cos_tilt_1'] = calc_cos1(
#         parameters["q"], parameters["a_1"], parameters["a_2"], parameters["xeff"], parameters["cos_tilt_2"])
#     converted_parameters['lim'] = cos_lim
    return converted_parameters


def get_samples(num=10):
    xeffs = np.linspace(-1, 1, 300)
    p_xeff = get_marginalised_chi_eff(xeffs)
    priors = PriorDict(conversion_function=convert_a1_a2_xeff_to_z)
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a_1'] = Uniform(minimum=0, maximum=1)
    priors['a_2'] = Uniform(minimum=0, maximum=1)
    priors['cos_tilt_2'] = Uniform(minimum=-1, maximum=1)
    priors['xeff'] = Interped(
        xeffs, p_xeff, minimum=-1, maximum=1, name='xeff', latex_label="$\chi_{\rm eff}$")
    priors['xeff_constraint'] = Constraint(minimum=0, maximum=1)
    priors['cos_tilt_1'] = Constraint(minimum=-1, maximum=1)
    s = pd.DataFrame(priors.sample(num))
    s['cos_tilt_1'] = calc_cos1(s.q, s.a_1, s.a_2, s.xeff, s.cos_tilt_2)
    s['valid_cos1'] = (np.abs(s['cos_tilt_1']) < 1) & (np.abs(s['xeff']) < 1)
    s['true_xeff'] = calculate_xeff(
        s.a_1, s.a_2, s.cos_tilt_1, s.cos_tilt_2, s.q)
    

    return s

def check_samples(s):
    s['cos_tilt_1'] = calc_cos1(s.q, s.a_1, s.a_2, s.xeff, s.cos_tilt_2)
    s['valid_cos1'] = (np.abs(s['cos_tilt_1']) < 1) & (np.abs(s['xeff']) < 1)
    s['true_xeff'] = calculate_xeff(s.a_1, s.a_2, s.cos_tilt_1, s.cos_tilt_2, s.q)
    s['diff'] = s['true_xeff'] - s['xeff']
    if (sum(s['valid_cos1'])!=len(s['valid_cos1'])):
        print(f"Samples have invalid cos1:")
        return s[s['valid_cos1']==False] 
    if (sum(s['diff'])>0.1):
        return s[s['diff']<=0.1]

s = get_samples()
check_samples(s)
# -


s = get_samples(10000)
fig = overlaid_corner(
    samples_list=[s],
    sample_labels=["Valid Samples"],
    params=['q', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'xeff'],
    samples_colors=['orange'], quants=True,
)


# ## Method 2: rejection sampling
#
# 1. Sample q, a1, a2, xeff, cos2
# 2. reject invalid xeff
# 3. reject invalid cos2
# 4. Calc cos1 and check xeff(q,a1,a2,cos1,cos2) = xeff

# +
@np.vectorize
def cos2_lower_lim(q, a1, a2, xeff):
    return min(calc_cos2_limits(q, a1, a2, xeff))

@np.vectorize
def cos2_upper_lim(q, a1, a2, xeff):
    return max(calc_cos2_limits(q, a1, a2, xeff))

def get_sample_in_lim(lim, pri):
    og_min, og_max = pri.minimum, pri.maximum
    pri.minimum,pri.maximum = lim[0], lim[1]
    s = pri.sample(1)
    pri.minimum, pri.maximum = og_min, og_max

#     print(lim)
#     s =  pri.sample(1)
#     while not (lim[0]<= s <= lim[1]):
#         s =  pri.sample(1)
    return s

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
    s.pop("valid_xeff")
    return s

@np.vectorize
def get_valid_xeff_samples(a1,a2, q, xeff_prior):
    """|xeff|<=(a1+qa2)/(q+1) [ie when CosT = -1 or 1]"""
    xeff_lim = (a1+a2*q)/(q+1)
    return get_sample_in_lim([-xeff_lim, xeff_lim], xeff_prior)

@np.vectorize
def get_valid_cos2_samples(a1,a2, q, xeff, cos_prior):
    """-1 <= (xeff(q+1)-a1)/(qa2) <= cos2 <= (xeff(q+1)+a1)/(qa2) <= 1"""
    lower = cos2_lower_lim(q, a1, a2, xeff)
    upper = cos2_upper_lim(q, a1, a2, xeff)
    return get_sample_in_lim([lower, upper], cos_prior)

@np.vectorize
def get_valid_xeff(q, a1, a2):
    xeffs = np.linspace(-1, 1, 300)
    x_p = chi_p_prior_from_isotropic_spins(q=q, aMax=np.maximum(a1, a2), xs=xeffs)
    xeff_prior = Interped(
        xeffs, p_xeff, minimum=-1, maximum=1, name='xeff', latex_label="$\chi_{\rm eff}$")
    return xeff_prior.sample(1)
    

def get_samples(num=10):
    priors = PriorDict()
    priors['q'] = Uniform(minimum=0, maximum=1)
    priors['a_1'] = Uniform(minimum=0, maximum=1)
    priors['a_2'] = Uniform(minimum=0, maximum=1)
    s = pd.DataFrame(priors.sample(num))

#     xeffs = np.linspace(-1, 1, 300)
#     p_xeff = get_marginalised_chi_eff(xeffs) 
#     xeff_prior = Interped(
#         xeffs, p_xeff, minimum=-1, maximum=1, name='xeff', latex_label="$\chi_{\rm eff}$")
    s['xeff'] = get_valid_xeff(s.a_1, s.a_1, s.q)
    s['xeff_lim'] = (s['a_1']+s['q']*s['a_2']) / (s['q']+1)
    
    cos_prior = Uniform(minimum=-1, maximum=1)
    s['cos_tilt_2'] = get_valid_cos2_samples(s.a_1, s.a_2, s.q, s.xeff, cos_prior)
    
    s['cos_tilt_1'] = calc_cos1(s.q, s.a_1, s.a_2, s.xeff, s.cos_tilt_2)
    s['lower_lims'] =  cos2_lower_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    s['upper_lims'] = cos2_upper_lim(q=s['q'], a1=s['a_1'], a2=s['a_2'], xeff=s['xeff'])
    return s


s = get_samples()
# check_samples(s)


# +
def plot_cos_given_other_params(q, a1,a2,xeff):
    cos_lims = calc_cos2_limits(q, a1, a2, xeff)
    cos_2_vals = np.linspace(-1, 1, num=300)
    cos_1_vals = calc_cos1(q, a1, a2, xeff, cos_2_vals)
    fig, ax = plt.subplots()
    ax.plot(cos_2_vals, cos_1_vals, 'k')
    ylims = min(cos_1_vals), max(cos_1_vals)
    ax.axhspan(min(ylims), -1,  alpha=0.5, color='orange')
    ax.axhspan(1, max(ylims),  alpha=0.5, color='orange')
    ax.axvspan(-1, cos_lims[0],  alpha=0.5, color='red')
    ax.axvspan(cos_lims[1], 1, alpha=0.5, color='red')
    ax.set_ylabel(r"$\cos\theta1$")
    ax.set_xlabel(r"$\cos\theta2$")
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
#     ax.set_ylim(*ylims)
    label= (f"$q={q:.1f}$" + "\n"
            f"$a_1={a1:.1f}$"+ "\n" 
            f"$a_2={a2:.1f}$"+"\n"
            f"$\chi_e={xeff:.1f}$"+"\n"
            f"$lim=[{cos_lims[0]:.1f},{cos_lims[1]:.1f}]%$"
           )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    plt.show()
    
plot_cos_given_other_params(q=0.9, a1=0.2, a2=0.9, xeff=-0.1)
# -

plot_interactive = False
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

s = get_samples(10000)

# +
import corner
params=['q', 'a_1', 'a_2', 'xeff', 'cos_tilt_2', 'cos_tilt_1']
plot_range = [(0,1), (0,1), (0,1), (-1,1), (-1,1), (-1,1)]
defaults_kwargs = dict(
            bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3, hist_kwargs=dict(density=True))


fig = corner.corner(s[params], **defaults_kwargs, range=plot_range)


priors = PriorDict()
priors['q'] = Uniform(minimum=0, maximum=1)
priors['a_1'] = Uniform(minimum=0, maximum=1)
priors['a_2'] = Uniform(minimum=0, maximum=1)
xeffs = np.linspace(-1, 1, 300)
p_xeff = get_marginalised_chi_eff(xeffs) 
priors['xeff'] = Interped(
    xeffs, p_xeff, minimum=-1, maximum=1, name='xeff', latex_label="$\chi_{\rm eff}$")
priors['cos_tilt_2'] = Uniform(minimum=-1, maximum=1)


axes = fig.get_axes()
for i, par in enumerate(params):
    ax = axes[i + i * len(params)]
    theta = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
    if par in priors:
        print(f"{i + i * len(params)} Plotting {par} for {ax.get_xlim()[0], ax.get_xlim()[1]}")
        ax.plot(theta, priors[par].prob(theta), color='red')

plt.savefig("TEST.png")

# -


