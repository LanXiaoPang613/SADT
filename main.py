"""Train SADT on CIFAR-10 or CIFAR-100 under noisy labels."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import build_loss, resolve_training_config
from datasets.data_loaders import HUMAN_LABEL_KEYS, build_loaders
from models import CIFAR10CNN, resnet34
from utils import evaluate, make_logger, seed_everything

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SADT reproduction on CIFAR-10/CIFAR-100"
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--root", required=True, help="CIFAR data root")
    parser.add_argument(
        "--noise-type",
        "--noise_type",
        dest="noise_type",
        choices=("symmetric", "asymmetric", "dependent", "human"),
        default="symmetric",
    )
    parser.add_argument(
        "--noise-rate",
        "--noise_rate",
        dest="noise_rate",
        default="0.0",
        help="numeric rate, or a CIFAR-N key for --noise-type human",
    )
    parser.add_argument(
        "--noise-method",
        "--noise_method",
        dest="noise_method",
        choices=("method1", "method2"),
        default="method2",
    )
    parser.add_argument("--human-label-path", default=None)
    parser.add_argument("--gpu", "--gpus", dest="gpu", default="0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", "--loss_alpha", dest="alpha", type=float, default=None)
    parser.add_argument("--beta", "--loss_beta", dest="beta", type=float, default=None)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments and print resolved paper settings without loading data",
    )
    return parser


def resolve_cifar_root(root: str, dataset: str) -> str:
    root_path = Path(root).expanduser().resolve()
    extracted = {
        "cifar10": "cifar-10-batches-py",
        "cifar100": "cifar-100-python",
    }[dataset]
    if (root_path / extracted).is_dir():
        return str(root_path)
    nested = root_path / dataset
    return str(nested if nested.exists() else root_path)


def validate_noise(dataset: str, noise_type: str, noise_rate: str) -> None:
    if noise_type == "human":
        allowed = (
            {"clean", "worst", "aggre", "rand1", "rand2", "rand3"}
            if dataset == "cifar10"
            else {"clean100", "noisy100"}
        )
        if noise_rate not in allowed:
            raise ValueError(f"CIFAR-N key must be one of {sorted(allowed)}")
        return
    rate = float(noise_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("noise rate must be in [0, 1]")
    if noise_type == "dependent" and rate not in {0.2, 0.4, 0.6}:
        raise ValueError("dependent noise ships exact index configs for 0.2, 0.4, 0.6")


def train_trial(args, trial: int, output_dir: Path, logger, writer):
    seed = args.seed + trial
    seed_everything(seed)
    config = resolve_training_config(
        args.dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=args.alpha,
        beta=args.beta,
    )
    if args.dataset == "cifar10":
        model = CIFAR10CNN()
    else:
        model = resnet34(classes=100)
    device = torch.device("cuda")
    model = model.to(device)
    criterion = build_loss(config)
    train_loader, test_loader = build_loaders(args, config.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.l2_weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)
    run_epochs = config.epochs if args.stop_after is None else args.stop_after
    if run_epochs <= 0 or run_epochs > config.epochs:
        raise ValueError("--stop-after must be in [1, configured epochs]")
    if args.save_final_model and run_epochs != config.epochs:
        raise ValueError("--save-final-model requires the complete schedule")

    curve_path = output_dir / f"acc_seed{seed}.log"
    best = 0.0
    last = 0.0
    with curve_path.open("w", encoding="utf-8") as curve:
        for epoch in tqdm(range(run_epochs), desc=f"SADT {args.dataset}", ncols=70):
            model.train()
            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                if config.l1_weight_decay:
                    loss = loss + config.l1_weight_decay * sum(
                        parameter.abs().sum() for parameter in model.parameters()
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            last, top5 = evaluate(test_loader, model, device)
            best = max(best, last)
            curve.write(f"{last}\n")
            curve.flush()
            writer.add_scalar(f"seed{seed}/test_top1", last, epoch + 1)
            logger.info(
                "seed=%d epoch=%d loss=%.6f top1=%.6f top5=%.6f",
                seed,
                epoch + 1,
                total_loss,
                last,
                top5,
            )

    if args.save_final_model:
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "metadata": {
                    "method": "SADT",
                    "seed": seed,
                    "config": asdict(config),
                    "noise_type": args.noise_type,
                    "noise_rate": args.noise_rate,
                    "best_top1": best,
                    "last_top1": last,
                },
            },
            output_dir / f"model_seed{seed}_final.pth",
        )
    return {"seed": seed, "best_top1": best, "last_top1": last}


def main() -> None:
    args = build_parser().parse_args()
    validate_noise(args.dataset, args.noise_type, args.noise_rate)
    args.root = resolve_cifar_root(args.root, args.dataset)
    config = resolve_training_config(
        args.dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=args.alpha,
        beta=args.beta,
    )
    if args.dry_run:
        print(json.dumps({"method": "SADT", **asdict(config)}, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the released training protocol")
    torch.backends.cudnn.benchmark = True

    output_dir = (
        args.output_root
        / args.dataset
        / "SADT"
        / f"{args.noise_type}_{args.noise_rate}"
        / args.tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir / "result.log")
    logger.info("args=%s", vars(args))
    logger.info("resolved_config=%s", asdict(config))
    writer = SummaryWriter(log_dir=str(output_dir))
    results = [
        train_trial(args, trial, output_dir, logger, writer)
        for trial in range(args.trials)
    ]
    writer.close()
    last = np.asarray([result["last_top1"] for result in results]) * 100.0
    summary = {
        "method": "SADT",
        "config": asdict(config),
        "runs": results,
        "last_mean_percent": float(last.mean()),
        "last_std_percent": float(last.std()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("SADT last: %.2f+/-%.2f", last.mean(), last.std())


if __name__ == "__main__":
    main()
