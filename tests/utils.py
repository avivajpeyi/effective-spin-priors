import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Interped, Uniform
from bilby.core.prior import PriorDict

from effective_spins.conversions import (
    calculate_xeff,
    calculate_xp,
    calculate_xp_given_xeff,
)
from effective_spins.xeff_prior import get_marginalised_chi_eff


def get_traditional_prior():
    priors = PriorDict()
    priors["q"] = Uniform(minimum=0, maximum=1)
    priors["a1"] = Uniform(minimum=0, maximum=1)
    priors["a2"] = Uniform(minimum=0, maximum=1)
    priors["cos2"] = Uniform(minimum=-1, maximum=1)
    priors["cos1"] = Uniform(minimum=-1, maximum=1)
    return priors


def get_xeff_prior():
    priors = PriorDict()
    xeff = np.linspace(-1, 1, num=1000)
    p_xeff = get_marginalised_chi_eff(xeff)
    priors["xeff"] = Interped(xx=xeff, yy=p_xeff)
    return priors


def get_traditional_samples(num_samples=10 ** 5):
    priors = get_traditional_prior()
    s = pd.DataFrame(priors.sample(num_samples))
    s["sin1"] = np.sqrt(1 - s.cos1 ** 2)
    s["sin2"] = np.sqrt(1 - s.cos2 ** 2)
    s["tan1"] = s["sin1"] / s["cos1"]
    s["tan2"] = s["sin2"] / s["cos2"]
    s["xeff"] = calculate_xeff(s.a1, s.a2, s.cos1, s.cos2, s.q)
    s["xp"] = calculate_xp(s.a1, s.a2, s.q, s.sin1, s.sin2)
    s["xp|xeff"] = calculate_xp_given_xeff(
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
