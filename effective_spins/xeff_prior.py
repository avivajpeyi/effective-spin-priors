"""xeff prior obtained from Tom's paper"""
import numpy as np
from scipy.special import spence as PL


def Di(z):
    """
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html

    Inputs
    z: A (possibly complex) scalar or array

    Returns
    Array equivalent to PolyLog[2,z], as defined by Mathematica
    """

    return PL(1.0 - z + 0j)


def chi_effective_prior_from_isotropic_spins(q, aMax, xs):
    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(np.abs(xs), -1)

    # Set up various piecewise cases
    pdfs = np.ones(xs.size, dtype=complex) * (-1.0)
    caseZ = xs == 0
    caseA = (xs > 0) * (xs < aMax * (1.0 - q) / (1.0 + q)) * (xs < q * aMax / (1.0 + q))
    caseB = (xs < aMax * (1.0 - q) / (1.0 + q)) * (xs > q * aMax / (1.0 + q))
    caseC = (xs > aMax * (1.0 - q) / (1.0 + q)) * (xs < q * aMax / (1.0 + q))
    caseD = (
        (xs > aMax * (1.0 - q) / (1.0 + q))
        * (xs < aMax / (1.0 + q))
        * (xs >= q * aMax / (1.0 + q))
    )
    caseE = (xs > aMax * (1.0 - q) / (1.0 + q)) * (xs > aMax / (1.0 + q)) * (xs < aMax)
    caseF = xs >= aMax

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    pdfs[caseZ] = (1.0 + q) / (2.0 * aMax) * (2.0 - np.log(q))

    pdfs[caseA] = (
        (1.0 + q)
        / (4.0 * q * aMax ** 2)
        * (
            q
            * aMax
            * (
                4.0
                + 2.0 * np.log(aMax)
                - np.log(q ** 2 * aMax ** 2 - (1.0 + q) ** 2 * x_A ** 2)
            )
            - 2.0 * (1.0 + q) * x_A * np.arctanh((1.0 + q) * x_A / (q * aMax))
            + (1.0 + q)
            * x_A
            * (Di(-q * aMax / ((1.0 + q) * x_A)) - Di(q * aMax / ((1.0 + q) * x_A)))
        )
    )

    pdfs[caseB] = (
        (1.0 + q)
        / (4.0 * q * aMax ** 2)
        * (
            4.0 * q * aMax
            + 2.0 * q * aMax * np.log(aMax)
            - 2.0 * (1.0 + q) * x_B * np.arctanh(q * aMax / ((1.0 + q) * x_B))
            - q * aMax * np.log((1.0 + q) ** 2 * x_B ** 2 - q ** 2 * aMax ** 2)
            + (1.0 + q)
            * x_B
            * (Di(-q * aMax / ((1.0 + q) * x_B)) - Di(q * aMax / ((1.0 + q) * x_B)))
        )
    )

    pdfs[caseC] = (
        (1.0 + q)
        / (4.0 * q * aMax ** 2)
        * (
            2.0 * (1.0 + q) * (aMax - x_C)
            - (1.0 + q) * x_C * np.log(aMax) ** 2.0
            + (aMax + (1.0 + q) * x_C * np.log((1.0 + q) * x_C))
            * np.log(q * aMax / (aMax - (1.0 + q) * x_C))
            - (1.0 + q)
            * x_C
            * np.log(aMax)
            * (2.0 + np.log(q) - np.log(aMax - (1.0 + q) * x_C))
            + q * aMax * np.log(aMax / (q * aMax - (1.0 + q) * x_C))
            + (1.0 + q)
            * x_C
            * np.log((aMax - (1.0 + q) * x_C) * (q * aMax - (1.0 + q) * x_C) / q)
            + (1.0 + q)
            * x_C
            * (Di(1.0 - aMax / ((1.0 + q) * x_C)) - Di(q * aMax / ((1.0 + q) * x_C)))
        )
    )

    pdfs[caseD] = (
        (1.0 + q)
        / (4.0 * q * aMax ** 2)
        * (
            -x_D * np.log(aMax) ** 2
            + 2.0 * (1.0 + q) * (aMax - x_D)
            + q * aMax * np.log(aMax / ((1.0 + q) * x_D - q * aMax))
            + aMax * np.log(q * aMax / (aMax - (1.0 + q) * x_D))
            - x_D
            * np.log(aMax)
            * (
                2.0 * (1.0 + q)
                - np.log((1.0 + q) * x_D)
                - q * np.log((1.0 + q) * x_D / aMax)
            )
            + (1.0 + q)
            * x_D
            * np.log((-q * aMax + (1.0 + q) * x_D) * (aMax - (1.0 + q) * x_D) / q)
            + (1.0 + q)
            * x_D
            * np.log(aMax / ((1.0 + q) * x_D))
            * np.log((aMax - (1.0 + q) * x_D) / q)
            + (1.0 + q)
            * x_D
            * (Di(1.0 - aMax / ((1.0 + q) * x_D)) - Di(q * aMax / ((1.0 + q) * x_D)))
        )
    )

    pdfs[caseE] = (
        (1.0 + q)
        / (4.0 * q * aMax ** 2)
        * (
            2.0 * (1.0 + q) * (aMax - x_E)
            - (1.0 + q) * x_E * np.log(aMax) ** 2
            + np.log(aMax)
            * (
                aMax
                - 2.0 * (1.0 + q) * x_E
                - (1.0 + q) * x_E * np.log(q / ((1.0 + q) * x_E - aMax))
            )
            - aMax * np.log(((1.0 + q) * x_E - aMax) / q)
            + (1.0 + q)
            * x_E
            * np.log(((1.0 + q) * x_E - aMax) * ((1.0 + q) * x_E - q * aMax) / q)
            + (1.0 + q)
            * x_E
            * np.log((1.0 + q) * x_E)
            * np.log(q * aMax / ((1.0 + q) * x_E - aMax))
            - q * aMax * np.log(((1.0 + q) * x_E - q * aMax) / aMax)
            + (1.0 + q)
            * x_E
            * (Di(1.0 - aMax / ((1.0 + q) * x_E)) - Di(q * aMax / ((1.0 + q) * x_E)))
        )
    )

    pdfs[caseF] = 0.0

    # Deal with spins on the boundary between cases
    if np.any(pdfs == -1):
        boundary = pdfs == -1
        pdfs[boundary] = 0.5 * (
            chi_effective_prior_from_isotropic_spins(q, aMax, xs[boundary] + 1e-6)
            + chi_effective_prior_from_isotropic_spins(q, aMax, xs[boundary] - 1e-6)
        )

    return np.real(pdfs)


def get_marginalised_chi_eff(xs):
    """
    Function defining the marginalised prior p(chi_eff) corresponding to
    uniform, isotropic component spin priors, and assuming the maximum
    allowed dimensionless component spin magnitude aMax=1

    Inputs
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """
    qs = np.linspace(0, 1, 300)
    p_xeff = np.zeros(len(qs))
    for q in qs[1:]:  # skip over q=0
        p_xeff += np.array(chi_effective_prior_from_isotropic_spins(q, aMax=1, xs=xs))
    p_xeff = p_xeff / (len(qs) - 1)
    return p_xeff
