"""CIFAR loaders for clean, synthetic, instance-dependent and CIFAR-N labels."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from datasets.data_dependent.build_dataset import build_dataset_dependent
from datasets.data_human.build_dataset import build_dataset_human
from datasets.data_method1.build_dataset import build_dataset_method1
from datasets.data_method2.build_dataset import build_dataset_method2
from datasets.transformers import get_transforms


HUMAN_LABEL_KEYS = {
    "clean": "clean_label",
    "worst": "worse_label",
    "aggre": "aggre_label",
    "rand1": "random_label1",
    "rand2": "random_label2",
    "rand3": "random_label3",
    "clean100": "clean_label",
    "noisy100": "noisy_label",
}


def _worker_init(worker_id: int) -> None:
    # Matches the released loader behavior.
    np.random.seed(123 + worker_id)


def _human_label_path(args) -> Path:
    if args.human_label_path:
        path = Path(args.human_label_path)
    else:
        filename = (
            "CIFAR-10_human.pt"
            if args.dataset == "cifar10"
            else "CIFAR-100_human.pt"
        )
        path = Path(__file__).parent / "data_human" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"CIFAR-N labels not found: {path}. See README.md for placement."
        )
    return path


def build_loaders(args, batch_size: int):
    train_transform, test_transform = get_transforms(args.dataset)
    if args.noise_type == "dependent":
        train_dataset, test_dataset = build_dataset_dependent(
            args.dataset,
            args.root,
            args.noise_type,
            float(args.noise_rate),
            train_transform,
            test_transform,
        )
    elif args.noise_type in {"symmetric", "asymmetric"}:
        builder = (
            build_dataset_method1
            if args.noise_method == "method1"
            else build_dataset_method2
        )
        train_dataset, test_dataset = builder(
            args.dataset,
            args.root,
            args.noise_type,
            float(args.noise_rate),
            train_transform,
            test_transform,
        )
    elif args.noise_type == "human":
        train_dataset, test_dataset = build_dataset_human(
            args.dataset,
            args.root,
            HUMAN_LABEL_KEYS[args.noise_rate],
            str(_human_label_path(args)),
            train_transform,
            test_transform,
        )
    else:
        raise ValueError(f"unsupported noise type: {args.noise_type}")

    persistent = args.workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=_worker_init,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, test_loader
