import numpy as np

from . import distribution_rules as dist
from .priors_conditional_on_xeff import a1_prior_given_xeff as p_a1
from .priors_conditional_on_xeff import a2_prior_given_xeff as p_a2
from .priors_conditional_on_xeff import cos2_prior_given_xeff as p_c2
from .priors_conditional_on_xeff import q_prior_given_xeff as p_q

N = 1000


def p_xp_given_xeff(xp, xeff, init_a1a2qcos2_prior):
    """
    xp = a1 sqrt(1 - (a+b)^2)

    where
     a = xeff(q+1)/a1
     b = a2qcos2/a1


    """
    #
    #
    # a2_
    #
    # a1, a2, c2, q = 0, 0, 0, 0
    # kwargs = dict(xeff=xeff, init_a1a2qcos2_prior=init_a1a2qcos2_prior)
    #
    # p_a2q = dist.product_distribution(z_vals=[], a_vals=[], pdf_a=foo, pdf_b=foo)
    # p_a2qc2 = dist.product_distribution(z_vals=[], a_vals=[], pdf_a=foo, pdf_b=foo)

    return 1


def p_a2q(a2q_vals, xeff, init_a1a2qcos2_prior):
    """
    a2q_vals in [0, 1]
    """
    kwargs = dict(xeff=xeff, init_a1a2qcos2_prior=init_a1a2qcos2_prior)
    q_vals = np.linspace(0, 1, N)
    return dist.product_distribution(
        a2q_vals, a_vals=q_vals, pdf_a=p_q, pdf_b=p_a2, kwargs_a=kwargs, kwargs_b=kwargs
    )


def p_a2qc2(a2qc2_vals, xeff, init_a1a2qcos2_prior):
    """
    a2qc2_vals in [-1, 1]
    """
    kwargs = dict(xeff=xeff, init_a1a2qcos2_prior=init_a1a2qcos2_prior)
    c2_vals = np.linspace(-1, 1, N)
    return dist.product_distribution(
        a2qc2_vals,
        a_vals=c2_vals,
        pdf_a=p_c2,
        pdf_b=p_a2q,
        kwargs_a=kwargs,
        kwargs_b=kwargs,
    )


def p_inverse_a1(a1_vals, xeff, init_a1a2qcos2_prior):
    kwargs = dict(xeff=xeff, init_a1a2qcos2_prior=init_a1a2qcos2_prior)
    return dist.inverse_distribution(a1_vals, p_a1, kwargs_a=kwargs)


def p_a2qc2_a1(a2qc2_a1_vals, xeff, init_a1a2qcos2_prior):
    pass
