"""Small training utilities with no loss-function side effects."""

import logging
import random
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        top5 = outputs.topk(5, dim=1).indices
        correct_top1 += top5[:, :1].eq(targets[:, None]).sum().item()
        correct_top5 += top5.eq(targets[:, None]).sum().item()
        total += targets.numel()
    return correct_top1 / total, correct_top5 / total


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(str(path.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
