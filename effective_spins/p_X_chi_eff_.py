# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pylab as plt
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib

# example


# # p_q(q|chi_eff)

# +
qs = np.linspace(0, 1, 1000)
xeffs = np.linspace(-1, 1, 1000)

p_q_xeff = np.zeros((len(qs), len(xeffs)))

i = 0
for i, _q in enumerate(qs):

    for j, _xeff in enumerate(xeffs):
        N_mcmc_samples = 10000
        # first sample a1, a2, c2:
        c2 = np.random.uniform(-1, 1, N_mcmc_samples)
        a1 = np.random.uniform(0, 1, N_mcmc_samples)
        a2 = np.random.uniform(0, 1, N_mcmc_samples)

        p_a1 = np.ones(N_mcmc_samples)
        p_a2 = np.ones(N_mcmc_samples)
        p_c2 = 0.5 * np.ones(N_mcmc_samples)
        p_q = np.ones(N_mcmc_samples)

        chi_min, chi_max = (
            -a1 / (1 + _q) + (a2 * _q * c2) / (1 + _q),
            a1 / (1 + _q) + (a2 * _q * c2) / (1 + _q),
        )

        p_xeff_given_a1a2c2 = stats.uniform.pdf(
            _xeff, loc=chi_min, scale=chi_max - chi_min
        )

        p_q_xeff[i][j] = (
            np.sum(p_a1 * p_a2 * p_c2 * p_q * p_xeff_given_a1a2c2) / N_mcmc_samples
        )
# -


# # p_a1(a1|chi_eff)

# +
a1s = np.linspace(0, 1, 1000)
xeffs = np.linspace(-1, 1, 1000)

p_a1_xeff = np.zeros((len(a1s), len(xeffs)))

i = 0
for i, _a1 in enumerate(a1s):

    for j, _xeff in enumerate(xeffs):
        N_mcmc_samples = 10000
        # first sample a1, a2, c2:
        c2 = np.random.uniform(-1, 1, N_mcmc_samples)
        q = np.random.uniform(0, 1, N_mcmc_samples)
        a2 = np.random.uniform(0, 1, N_mcmc_samples)

        p_a1 = np.ones(N_mcmc_samples)
        p_a2 = np.ones(N_mcmc_samples)
        p_c2 = 0.5 * np.ones(N_mcmc_samples)
        p_q = np.ones(N_mcmc_samples)

        chi_min, chi_max = (
            -a1 / (1 + _q) + (a2 * _q * c2) / (1 + _q),
            a1 / (1 + _q) + (a2 * _q * c2) / (1 + _q),
        )

        p_xeff_given_a1a2c2 = stats.uniform.pdf(
            _xeff, loc=chi_min, scale=chi_max - chi_min
        )

        p_a1_xeff[i][j] = (
            np.sum(p_a1 * p_a2 * p_c2 * p_q * p_xeff_given_a1a2c2) / N_mcmc_samples
        )
# -


X, Y = np.meshgrid(qs, xeffs)

plt.pcolor(X, Y, p_q_xeff, cmap="pink")
plt.xlabel("q")
plt.ylabel("chi_eff")
plt.show()

stats.uniform.pdf(_xeff, loc=-0.12866281519621883, scale=chi_min[3] + chi_max[3])

p_a1_xeff

p_q_xeff
