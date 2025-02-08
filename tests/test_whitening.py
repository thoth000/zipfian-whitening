import numpy as np
import torch

from src.zipfian_whitening import UniformWhitening, ZipfianWhitening


class TestUniformWhitening(object):
    def test_fit_transform(self):
        W = torch.rand(10, 5)
        p = torch.rand(10)
        uw = UniformWhitening()
        Wt = uw.fit_transform(W, p)
        assert Wt.shape == W.shape
        I = torch.eye(W.shape[1])
        assert torch.allclose(torch.cov(Wt.T), I, atol=1e-5)


class TestZipfianWhitening(object):
    def test_fit_transform(self):
        W = torch.rand(10, 5)
        p = torch.ones(10)
        p = p / p.sum()
        zw = ZipfianWhitening()
        Wt = zw.fit_transform(W, p)
        assert Wt.shape == W.shape
        I = torch.eye(W.shape[1])
        print(torch.cov(Wt.T))
        print(torch.cov(Wt.T)[0, 0])
        assert torch.allclose(torch.cov(Wt.T), I, atol=1e-4)
