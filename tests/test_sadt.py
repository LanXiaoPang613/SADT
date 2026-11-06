"""Formula and architecture smoke tests."""

import unittest

import torch

from config import PAPER_CONFIGS, build_loss
from models import CIFAR10CNN, resnet34


class SADTSmokeTest(unittest.TestCase):
    def test_class_sum_and_backward(self):
        for classes, dataset in ((10, "cifar10"), (100, "cifar100")):
            torch.manual_seed(123)
            logits = torch.randn(4, classes, requires_grad=True)
            targets = torch.tensor([0, 1, 2, 3])
            criterion = build_loss(PAPER_CONFIGS[dataset])
            classwise = criterion.classwise_loss(logits)
            expected = criterion.alpha + criterion.beta * (classes - 1)
            self.assertTrue(torch.allclose(
                classwise.sum(dim=1), torch.full((4,), expected),
                atol=2e-5, rtol=2e-5,
            ))
            loss = criterion(logits, targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_model_shapes(self):
        self.assertEqual(CIFAR10CNN()(torch.randn(2, 3, 32, 32)).shape, (2, 10))
        self.assertEqual(resnet34(100)(torch.randn(2, 3, 32, 32)).shape, (2, 100))


if __name__ == "__main__":
    unittest.main()
