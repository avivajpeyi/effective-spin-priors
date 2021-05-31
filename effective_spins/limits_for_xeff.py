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

import numpy as np
import pandas as pd
import sympy as sym
from ipywidgets import FloatSlider, interact
import bilby
from bilby.core.prior import Uniform
import  matplotlib.pyplot as plt
# %config InlineBackend.print_figure_kwargs = {'facecolor': "w"}
# %matplotlib inline

# # Limits for xeff parameters

# \begin{align}
# \newcommand{\xp}{{\chi_{\text{p}}}}
# \newcommand{\xeff}{{\chi_{\text{eff}}}}
#      \xeff = \frac{\chi_1\cos\theta_1 + q\chi_2\cos\theta_2}{1+q} 
# \end{align}

# +
def get_equations():
    xeff = sym.Symbol('xeff')
    a1 = sym.Symbol('a1')
    a2 = sym.Symbol('a2')
    cos1 = sym.Symbol('cos1')
    cos2 = sym.Symbol('cos2')
    q = sym.Symbol('q')
    xeff_eqn = sym.Eq(((a1 * cos1 + q * a2 * cos2) / (1 + q)) - xeff, 0)

    sols = []
    vrs = [xeff, a1, a2, cos1, cos2, q]
    for var in vrs:
        sols.append(sym.solveset(xeff_eqn, var))

    for v, sol in zip(vrs, sols):
        c = (
            f"""
        def calculate_{v}({",".join([f"{x}" for x in vrs])}):
            return {sol}
        """
        )
        c = c.replace("FiniteSet", "")
        c = c.replace("Complement(", "")
        c = c.replace(", (-1))", "")
        print(c)


get_equations()


# +
def calculate_xeff(xeff, a1, a2, cos1, cos2, q):
    return ((a1 * cos1 + a2 * cos2 * q) / (q + 1))


def calculate_a1(xeff, a1, a2, cos1, cos2, q):
    return (-(a2 * cos2 * q - xeff * (q + 1)) / cos1)


def calculate_a2(xeff, a1, a2, cos1, cos2, q):
    return (-(a1 * cos1 - xeff * (q + 1)) / (cos2 * q))


def calculate_cos1(xeff, a1, a2, cos1, cos2, q):
    return (-(a2 * cos2 * q - xeff * (q + 1)) / a1)


def calculate_cos2(xeff, a1, a2, cos1, cos2, q):
    return (-(a1 * cos1 - xeff * (q + 1)) / (a2 * q))


def calculate_q(xeff, a1, a2, cos1, cos2, q):
    return (-(a1 * cos1 - xeff) / (a2 * cos2 - xeff))

def q_factor(q):
    return ((3. + 4. * q) / (4. + 3. * q)) * q


def calculate_xp(xeff, a1, a2, cos1, cos2, q):
    sin1 , sin2 = np.sin(np.arccos(cos1)), np.sin(np.arccos(cos2))
    case1 = a1 * sin1
    case2 = a2 * sin2 * q_factor(q)
    return np.maximum(case1, case2)


def tabulate_calculated_values(xeff, a1, a2, cos1, cos2, q):
    xeff = np.float64(xeff)
    a1 = np.float64(a1)
    a2 = np.float64(a2)
    cos1 = np.float64(cos1)
    cos2 = np.float64(cos2)
    q = np.float64(q)
    d = dict(
        xp = calculate_xp(xeff, a1, a2, cos1, cos2, q),
        xeff=calculate_xeff(xeff, a1, a2, cos1, cos2, q),
        a1=calculate_a1(xeff, a1, a2, cos1, cos2, q),
        a2=calculate_a2(xeff, a1, a2, cos1, cos2, q),
        cos1=calculate_cos1(xeff, a1, a2, cos1, cos2, q),
        cos2=calculate_cos2(xeff, a1, a2, cos1, cos2, q),
        q=calculate_q(xeff, a1, a2, cos1, cos2, q)
    )
    df = pd.DataFrame()
    df = df.append(d, ignore_index=True)
    return df


# -


interact(
    tabulate_calculated_values,
    xeff=FloatSlider(min=-1, max=1, step=0.1, continuous_update=False),
    a1=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
    a2=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
    cos1=FloatSlider(min=-1, max=1, step=0.1, continuous_update=False),
    cos2=FloatSlider(min=-1, max=1, step=0.1, continuous_update=False),
    q=FloatSlider(min=0.1, max=1, step=0.1, continuous_update=False),
)


# +
def get_samples(num=1e4):
    s = pd.DataFrame(bilby.prior.PriorDict(dict(
        a1 = Uniform(0,1),
        a2 = Uniform(0,1),
        cos1 = Uniform(-1,1),
        cos2 = Uniform(-1,1),
        q = Uniform(0,1)
    )).sample(int(num)))
    s['xeff'] = calculate_xeff(_, s.a1, s.a2, s.cos1, s.cos2, s.q)
    s['xp'] = calculate_xp(_, s.a1, s.a2, s.cos1, s.cos2, s.q)
    s['s1z'] = s.cos1 * s.a1
    s['s2z'] = s.cos2 * s.a2
    return s

s = get_samples()


# +
def plot_relation(k, xlim, draw_limits=None):
    fig, ax = plt.subplots()
    ax.scatter(s[k], s.xeff, color='tab:orange', alpha=0.1, marker=",")  
    ax.set_xlim(xlim)
    ax.set_ylim(-1,1)
    ax.set_xlabel(k)
    ax.set_ylabel("xeff")
    plt.show()
    
plot_relation('a1', (-1,1))
plot_relation('s2z', (-1,1))
plot_relation('q', (0,1))
# -



# ## Draw limits -0.5<=xeff<=0.5

# + pycharm={"name": "#%%\n"}
def filter_samples_for_xeff_middle_lim(s):
    samp = s.copy()
#     samp = samp[samp['xeff'] <= 0.5]
#     samp = samp[samp['xeff'] >= -0.5]
    return samp


# -

# ### a1 limit

# + pycharm={"name": "#%%\n"}
def get_a1_limit(xeff):
    return np.abs(xeff)

@np.vectorize
def is_a1_in_limit(xeff, a1):
    return -np.abs(xeff) <= a1 <= np.abs(xeff)


def plot_a1_samples(s):
    # only keep samples in xeff range
    samp = filter_samples_for_xeff_middle_lim(s)
    fig, ax = plt.subplots()
    limit_checks = is_a1_in_limit(samp['xeff'], samp['a1'])
    cols = ["green" if v else "red" for v in limit_checks]
    ax.scatter(np.abs(samp['xeff']), samp['xeff'])
    ax.scatter(-np.abs(samp['xeff']), samp['xeff'])
    ax.scatter(samp['a1'], samp['xeff'], color=cols, alpha=0.1, marker=",", label="samples")  
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('a1')
    ax.set_ylabel("xeff")
    ax.axhspan(0.5, 1, color='k', alpha=0.2 )
    ax.axhspan(-1, -0.5, color='k', alpha=0.2)
    lim_a1 = np.linspace(0, 1, num=500)
    ax.plot(lim_a1, get_a1_limit(lim_a1), 'red')
    ax.plot(lim_a1, -get_a1_limit(lim_a1),'red', label="limit")
    plt.legend()
    plt.show()
    samp['check'] = limit_checks
    return samp[['check', 'a1', 'xeff']]

plot_a1_samples(s)


# + pycharm={"name": "#%%\n"}
def get_q_lower_limit(xeff, a1):
    return (-xeff + a1)/(-1+xeff)

@np.vectorize
def is_q_in_limit(xeff, a1, q):
    return get_q_lower_limit(xeff, a1) <= q <= 1

def plot_q_samples(s):
    # only keep samples in xeff range
    samp = filter_samples_for_xeff_middle_lim(s)
    fig, ax = plt.subplots()
    limit_checks = is_q_in_limit(samp['xeff'], samp['a1'], samp['q'])
    cols = ["green" if v else "red" for v in limit_checks]
    ax.scatter(samp['q'], samp['xeff'], alpha=0.1, marker=",", color=cols, label="samples")  
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('q')
    ax.set_ylabel("xeff")
    ax.axhspan(0.5, 1, color='k', alpha=0.2 )
    ax.axhspan(-1, -0.5, color='k', alpha=0.2)
    qlower = get_q_lower_limit(samp['xeff'], samp['a1'])
    ax.scatter(qlower,samp['xeff'], 'red', label="lower limit")
    plt.legend()
    plt.show()

plot_q_samples(s)


# + pycharm={"name": "#%%\n"}
def get_q_lower_limit(xeff, a1):
    return (-xeff + a1)/(-1+xeff)

@np.vectorize
def is_a2_in_limit(xeff, a1, q):
    return get_q_lower_limit(xeff, a1) <= q <= 1

def plot_a2_samples(s):
    # only keep samples in xeff range
    samp = filter_samples_for_xeff_middle_lim(s)
    fig, ax = plt.subplots()
    limit_checks = is_q_in_limit(samp['xeff'], samp['a2'], samp['q'])
    cols = ["green" if v else "red" for v in limit_checks]
    ax.scatter(samp['q'], samp['xeff'], alpha=0.1, marker=",", color=cols, label="samples")  
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('a2')
    ax.set_ylabel("xeff")
    ax.axhspan(0.5, 1, color='k', alpha=0.2 )
    ax.axhspan(-1, -0.5, color='k', alpha=0.2)
    qlower = get_q_lower_limit(samp['xeff'], samp['a2'])
    ax.scatter(qlower,samp['xeff'], 'red', label="lower limit")
    plt.legend()
    plt.show()

plot_a2_samples(s)


# -

#

def filter_samples_for_xeff_middle_lim(s):
    samp = s.copy()
    samp = samp[samp['xeff'] <= 0.5]
    samp = samp[samp['xeff'] >= -0.5]
    return samp


# ### a1 limit

# +
def get_a1_limit(xeff):
    return np.abs(xeff)

@np.vectorize
def is_a1_in_limit(xeff, a1):
    return -np.abs(xeff) <= a1 <= np.abs(xeff)


def plot_a1_samples(s):
    # only keep samples in xeff range
    samp = filter_samples_for_xeff_middle_lim(s)
    fig, ax = plt.subplots()
    limit_checks = is_a1_in_limit(samp['xeff'], samp['a1'])
    cols = ["green" if v else "red" for v in limit_checks]
    ax.scatter(np.abs(samp['xeff']), samp['xeff'])
    ax.scatter(-np.abs(samp['xeff']), samp['xeff'])
    ax.scatter(samp['a1'], samp['xeff'], color=cols, alpha=0.1, marker=",", label="samples")  
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('a1')
    ax.set_ylabel("xeff")
    ax.axhspan(0.5, 1, color='k', alpha=0.2 )
    ax.axhspan(-1, -0.5, color='k', alpha=0.2)
    lim_a1 = np.linspace(0, 1, num=500)
    ax.plot(lim_a1, get_a1_limit(lim_a1), 'red')
    ax.plot(lim_a1, -get_a1_limit(lim_a1),'red', label="limit")
    plt.legend()
    plt.show()
    samp['check'] = limit_checks
    return samp[['check', 'a1', 'xeff']]

plot_a1_samples(s)


# +
def get_q_lower_limit(xeff, a1):
    return (-xeff + a1)/(-1+xeff)

@np.vectorize
def is_q_in_limit(xeff, a1, q):
    return get_q_lower_limit(xeff, a1) <= q <= 1

def plot_q_samples(s):
    # only keep samples in xeff range
    samp = filter_samples_for_xeff_middle_lim(s)
    fig, ax = plt.subplots()
    limit_checks = is_q_in_limit(samp['xeff'], samp['a1'], samp['q'])
    cols = ["green" if v else "red" for v in limit_checks]
    ax.scatter(samp['q'], samp['xeff'], alpha=0.1, marker=",", color=cols, label="samples")  
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('q')
    ax.set_ylabel("xeff")
    ax.axhspan(0.5, 1, color='k', alpha=0.2 )
    ax.axhspan(-1, -0.5, color='k', alpha=0.2)
    qlower = get_q_lower_limit(samp['xeff'], samp['a1'])
    ax.scatter(qlower,samp['xeff'], 'red', label="lower limit")
    plt.legend()
    plt.show()

plot_q_samples(s)


# +
def get_q_lower_limit(xeff, a1):
    return (-xeff + a1)/(-1+xeff)

@np.vectorize
def is_a2_in_limit(xeff, a1, q):
    return get_q_lower_limit(xeff, a1) <= q <= 1

def plot_a2_samples(s):
    # only keep samples in xeff range
    samp = filter_samples_for_xeff_middle_lim(s)
    fig, ax = plt.subplots()
    limit_checks = is_q_in_limit(samp['xeff'], samp['a2'], samp['q'])
    cols = ["green" if v else "red" for v in limit_checks]
    ax.scatter(samp['q'], samp['xeff'], alpha=0.1, marker=",", color=cols, label="samples")  
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('a2')
    ax.set_ylabel("xeff")
    ax.axhspan(0.5, 1, color='k', alpha=0.2 )
    ax.axhspan(-1, -0.5, color='k', alpha=0.2)
    qlower = get_q_lower_limit(samp['xeff'], samp['a2'])
    ax.scatter(qlower,samp['xeff'], 'red', label="lower limit")
    plt.legend()
    plt.show()

plot_a2_samples(s)

# +
import corner
from agn_utils.plotting.overlaid_corner_plotter import CORNER_KWARGS
def plot_xeff_xp(s):
    corner.corner(s[['a1','a2','q', 'xeff']],**CORNER_KWARGS, color='tab:blue', range=[(0,1),(0,1),(0,1),(-1,1)])
    plt.show()

plot_xeff_xp(s)

# + pycharm={"name": "#%%\n"}

