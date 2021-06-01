from typing import Callable

import numpy as np


def product_distribution(
    z_vals: np.array,
    a_vals: np.array,
    pdf_a: Callable,
    pdf_b: Callable,
    kwargs_a={},
    kwargs_b={},
):
    """
    f_{Z}(z) = \int f_{A}(a) f_{B}(z/a) 1/|a| da
    """
    fz = np.zeros(len(z_vals))
    for i, z in enumerate(z_vals):
        fa = pdf_a(a_vals, **kwargs_a)
        fb = pdf_b(z / a_vals, **kwargs_b)
        da = a_vals[1] - a_vals[0]
        fz[i] = np.sum(fa * fb * (1 / np.abs(a_vals)) * da)
    return fz


def sum_distribution(
    z_vals: np.array,
    a_vals: np.array,
    pdf_a: Callable,
    pdf_b: Callable,
    kwargs_a={},
    kwargs_b={},
):
    """
    f_{Z}(z) = \sum_{a\inA} f_{A}(a) f_{B}(z-a) da
    """
    fz = np.zeros(len(z_vals))
    for i, z in enumerate(z_vals):
        fa = pdf_a(a_vals, **kwargs_a)
        fb = pdf_b(z - a_vals, **kwargs_b)
        da = a_vals[1] - a_vals[0]
        fz[i] = np.sum(fa * fb * da)
    return fz


def inverse_distribution(a_vals, pdf_a, kwargs_a={}):
    """
    z=1/a
    f_{Z}(a) = 1/(a^2) * f_{A}
    """
    return (1.0 / a_vals ** 2) * pdf_a(1.0 / a_vals, **kwargs_a)
