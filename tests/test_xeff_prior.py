import os
import shutil
import unittest

from effective_spins import xeff_prior
from . import utils


class TestXeffPrior(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/xeff_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_prior_on_xeff(self):
        samples = utils.get_traditional_samples()
        fig = utils.plot_funct_and_samples(
            xeff_prior.get_marginalised_chi_eff,
            samples.xeff,
            [-1, 1],
            [r"$\chi_{\rm eff}$", r"$p(\chi_{\rm eff})$"],
        )
        fig.savefig(f"{self.outdir}/p_xeff.png")


if __name__ == "__main__":
    unittest.main()
