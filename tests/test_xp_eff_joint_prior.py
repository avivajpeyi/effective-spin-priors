import os
import shutil
import unittest

from effective_spins import joint_xp_xeff_prior


class TestXpXeffJointPrior(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/xeff_testoutdir"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_prior_on_xeff(self):
        val = joint_xp_xeff_prior.p_xp_xeff(xp=0.1, xeff=1)
        self.assertNotEqual(1, val)


if __name__ == "__main__":
    unittest.main()
