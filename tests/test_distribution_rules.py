import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import Normal, Uniform

from effective_spins import distribution_rules


class TestDistributionRules(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/distribution_outdir"
        os.makedirs(self.outdir, exist_ok=True)
        self.N = 10000

    def uniform_distribution(self, min=-1, max=1):
        dist = Uniform(minimum=min, maximum=max)
        samples = dist.sample(self.N)
        pdf = dist.prob(samples)
        return samples, pdf, dist

    def normal_distribution(self, mu=0, sigma=1):
        dist = Normal(mu=mu, sigma=sigma)
        samples = dist.sample(self.N)
        pdf = dist.prob(samples)
        return samples, pdf, dist

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_sum_rule(self):
        a, pdf_a, dist_a = self.uniform_distribution()
        b, pdf_b, dist_b = self.uniform_distribution(0, 1)
        samples_c = a + b
        a_vals = np.linspace(-1, 1, self.N)
        z_vals = np.linspace(-1, 2, self.N)
        pdf_c = distribution_rules.sum_distribution(
            z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
        )
        plot_distribution(
            samples_c, pdf_c, z_vals, f"{self.outdir}/sum.png", title="Sum of Unif"
        )

    def test_product_rule(self):
        a, pdf_a, dist_a = self.uniform_distribution()
        b, pdf_b, dist_b = self.uniform_distribution(0, 1)
        a_vals = np.linspace(-1, 1, self.N)
        z_vals = np.linspace(-1, 1, self.N)
        samples_c = a * b
        pdf_c = distribution_rules.product_distribution(
            z_vals=z_vals, a_vals=a_vals, pdf_a=dist_a.prob, pdf_b=dist_b.prob
        )
        plot_distribution(
            samples_c, pdf_c, z_vals, f"{self.outdir}/prod.png", title="Prod of Unif"
        )

    def test_inverse_rule(self):
        a, pdf_a, dist_a = self.uniform_distribution()
        a_vals = np.linspace(-10, 10, self.N)
        samples_c = 1.0 / a
        pdf_c = distribution_rules.inverse_distribution(
            a_vals=a_vals, pdf_a=dist_a.prob
        )
        plot_distribution(
            samples_c,
            pdf_c,
            a_vals,
            f"{self.outdir}/inverse.png",
            title="Inverse of unif",
            bins=np.linspace(-10, 10, 100),
        )


def plot_distribution(samples, pdf, xvals, fname, title="", bins=50):
    plt.hist(samples, density=True, label="Hist(samples) PDF", bins=bins)
    plt.plot(xvals, pdf, label="Numerical PDF")
    plt.xlim(min(xvals), max(xvals))
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.legend()
    plt.suptitle(title)
    plt.savefig(fname)
    plt.close("all")


if __name__ == "__main__":
    unittest.main()
