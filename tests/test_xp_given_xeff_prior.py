import os
import shutil
import unittest

from effective_spins import xp_given_xeff
from . import utils


class TestXpGivenXeffPrior(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/xp_given_xeff_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_xp_given_xeff_prior(self):
        val = xp_given_xeff.p_xp_given_xeff(xp=0.1, xeff=1)
        self.assertNotEqual(1, val)


if __name__ == "__main__":
    unittest.main()
