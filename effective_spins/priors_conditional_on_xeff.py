"""a1, a2, q, theta1, theta2 conditional on xeff"""
import numpy as np
from scipy import stats
import pandas as pd

MCMC_SAMPLES = 10000


def _xeff_min_max(a1, a2, q, c2):
    xeff_lim = a1 / (1 + q) + (a2 * q * c2) / (1 + q)
    return -xeff_lim, xeff_lim


def _p_xeff_given_a1a2qc2(xeff, a1, a2, q, c2):
    xeff_min, xeff_max = _xeff_min_max(a1, a2, q, c2)
    p_xeff_given_a1a2qc2 = stats.uniform.pdf(
        xeff, loc=xeff_min, scale=xeff_max - xeff_min
    )
    return p_xeff_given_a1a2qc2


def _p_a1a2qc2(a1, a2, q, c2, init_a1a2qcos2_prior):
    p_a1 = init_a1a2qcos2_prior["a1"].prob(a1)
    p_a2 = init_a1a2qcos2_prior["a2"].prob(a2)
    p_c2 = init_a1a2qcos2_prior["cos2"].prob(c2)
    p_q = init_a1a2qcos2_prior["q"].prob(q)
    return p_a1 * p_a2 * p_c2 * p_q


@np.vectorize
def _p_param_given_xeff(param, xeff, init_a1a2qcos2_prior, param_key):
    s = pd.DataFrame(init_a1a2qcos2_prior.sample(MCMC_SAMPLES))
    s[param_key] = param
    p_xeff_given_a1a2qc2 = _p_xeff_given_a1a2qc2(xeff, s.a1, s.a2, s.q, s.cos2)
    p_a1a2qc2 = _p_a1a2qc2(s.a1, s.a2, s.q, s.cos2, init_a1a2qcos2_prior)
    return np.sum(p_a1a2qc2 * p_xeff_given_a1a2qc2) / MCMC_SAMPLES


def a1_prior_given_xeff(a1, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(a1, xeff, init_a1a2qcos2_prior, "a1")


def a2_prior_given_xeff(a2, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(a2, xeff, init_a1a2qcos2_prior, "a2")


def q_prior_given_xeff(q, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(q, xeff, init_a1a2qcos2_prior, "q")


def cos2_prior_given_xeff(cos2, xeff, init_a1a2qcos2_prior):
    return _p_param_given_xeff(cos2, xeff, init_a1a2qcos2_prior, "cos2")
