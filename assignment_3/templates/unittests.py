import unittest
import numpy as np
import torch

from a3_nf_template import Coupling, Flow, get_mask


def mean_error(x, y):
    return np.mean(np.abs(x - y))


def mean_rel_error(x, y):
    return np.mean(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def f_layer(layer, x, logdet):
    with torch.no_grad():
        z, logdet = layer(x, logdet, reverse=False)
        recon, logdet = layer(z, logdet, reverse=True)

    x, recon, logdet = x.numpy(), recon.numpy(), logdet.numpy()

    return x, recon, logdet


class TestLayers(unittest.TestCase):

    def test_flow(self):
        np.random.seed(42)
        error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            C = 784
            x = torch.randn(N, C)
            logdet = torch.zeros(N)

            layer = Flow([C], n_flows=2)

            x, recon, logdet = f_layer(layer, x, logdet)

            self.assertLess(mean_rel_error(x, recon), error_max)
            self.assertLess(mean_error(logdet, np.zeros(N)), error_max)

    def test_coupling(self):
        np.random.seed(42)
        error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            C = 784
            x = torch.randn(N, C)
            logdet = torch.zeros(N)

            layer = Coupling(c_in=C, mask=get_mask())

            x, recon, logdet = f_layer(layer, x, logdet)

            self.assertLess(mean_rel_error(x, recon), error_max)
            self.assertLess(mean_error(logdet, np.zeros(N)), error_max)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
    unittest.TextTestRunner(verbosity=2).run(suite)
