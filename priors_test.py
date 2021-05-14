import os
import shutil
import unittest

import numpy as np

from effective_spins import priors


class TestIsotropicSpins(unittest.TestCase):

    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_something(self):
        ans = priors.chi_p_prior_from_isotropic_spins(q=1, aMax=1, xs=0)
        self.assertIsInstance(ans, type(np.array([])))
        self.assertEqual(ans[0], 0.)


def test_me_as_well():
    print("yayaya")
    assert True


def test_me_as_well2():
    assert False


def test_me_as_well3():
    assert True


if __name__ == '__main__':
    unittest.main()
