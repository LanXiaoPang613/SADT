"""Standalone implementation of SADT.

This module contains only the proposed loss and has no dependency on the
benchmark configuration code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SADTLoss(nn.Module):
    """Surprise-Annealed Dual-Tail Loss (SADT).

    Args:
        alpha: Weight of the normalized active branch.
        beta: Weight of the log-balanced passive branch.
        eps: Numerical floor used by the NNCE evidence map.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not math.isfinite(eps) or not 0 < eps < 1:
            raise ValueError("eps must be finite and in (0, 1)")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    @staticmethod
    def _probability_dtype(logits: torch.Tensor) -> torch.dtype:
        if logits.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return logits.dtype

    @staticmethod
    def _joined_tail(surprise: torch.Tensor, chance: float) -> torch.Tensor:
        """C1 asinh-to-atan continuation joined at excess surprise log(K)."""
        excess = (surprise - chance).clamp_min(0.0)
        num_classes = float(round(math.exp(chance)))
        scale = math.sqrt(num_classes)
        join = chance

        lower = scale * torch.asinh(excess / scale)
        join_value = scale * math.asinh(join / scale)
        join_slope = 1.0 / math.sqrt(1.0 + (join / scale) ** 2)
        upper = join_value + join_slope * join * torch.atan(
            (excess - join) / join
        )
        transformed = chance + torch.where(excess <= join, lower, upper)
        return torch.where(surprise <= chance, surprise, transformed)

    def classwise_components(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return active and passive losses for every candidate class."""
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("logits must have shape [batch, classes] with classes >= 2")

        dtype = self._probability_dtype(logits)
        log_prob = F.log_softmax(logits.to(dtype=dtype), dim=1)
        probabilities = log_prob.exp()
        num_classes = logits.shape[1]
        chance = math.log(num_classes)
        surprise = -log_prob

        robust_surprise = self._joined_tail(surprise, chance)
        anchor = chance / surprise.clamp_min(chance)
        annealed = anchor * surprise + (1.0 - anchor) * robust_surprise
        annealed = torch.where(surprise <= chance, surprise, annealed)
        active = annealed / annealed.sum(dim=1, keepdim=True).clamp_min(self.eps)

        log_floor = math.log(self.eps)
        evidence = -log_floor + log_prob.clamp_min(log_floor)
        nnce = 1.0 - evidence / evidence.sum(dim=1, keepdim=True).clamp_min(self.eps)
        mae = 1.0 - probabilities
        mae_weight = 1.0 / math.log(num_classes)
        passive = (1.0 - mae_weight) * nnce + mae_weight * mae
        return active, passive

    def classwise_loss(self, logits: torch.Tensor) -> torch.Tensor:
        active, passive = self.classwise_components(logits)
        return self.alpha * active + self.beta * passive

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise ValueError("target must have shape [batch]")
        return self.classwise_loss(logits).gather(1, target[:, None]).mean()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, eps={self.eps}"
