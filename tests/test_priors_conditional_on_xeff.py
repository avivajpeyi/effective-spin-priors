import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from effective_spins import priors_conditional_on_xeff
from . import utils


class TestPriorConditionalOnXeff(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/conditional_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)
        self.p = utils.get_traditional_prior()
        self.xeffs = [-0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75]
        self.N = 100

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def plot_conditional_prior(self, param, xeffs, param_key):
        fig, ax = plt.subplots()
        for xeff in tqdm(xeffs, desc=f"p({param_key}|xeff) "):
            p_param = priors_conditional_on_xeff._p_param_given_xeff(
                param=param,
                xeff=xeff,
                init_a1a2qcos2_prior=self.p,
                param_key=param_key,
            )
            ax.plot(param, p_param, "-", label=f"xeff={xeff}")
        ax.set_ylabel(f"p({param_key}|xeff)")
        ax.set_xlabel(param_key)
        ax.set_xlim(min(param), max(param))
        ax.legend(loc="upper right")
        plt.savefig(f"{self.outdir}/p_{param_key}_given_xeff.png")

    def test_a1_conditional_priors(self):
        a1 = np.linspace(0, 1, self.N)
        self.plot_conditional_prior(param=a1, xeffs=self.xeffs, param_key="a1")

    def test_q_conditional_priors(self):
        q = np.linspace(0, 1, self.N)
        self.plot_conditional_prior(param=q, xeffs=self.xeffs, param_key="q")

    def test_a2_conditional_priors(self):
        a2 = np.linspace(0, 1, self.N)
        self.plot_conditional_prior(param=a2, xeffs=self.xeffs, param_key="a2")

    def test_cos2_conditional_priors(self):
        cos2 = np.linspace(-1, 1, self.N)
        self.plot_conditional_prior(param=cos2, xeffs=self.xeffs, param_key="cos2")

    def test_simple_val(self):
        p_a1 = priors_conditional_on_xeff.a1_prior_given_xeff(
            a1=0.1, xeff=0.2, init_a1a2qcos2_prior=self.p
        )
        self.assertNotEqual(p_a1, 1)


if __name__ == "__main__":
    unittest.main()
