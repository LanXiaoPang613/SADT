"""Run the three-seed SADT CIFAR experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENARIOS = {
    "clean": ("symmetric", "0.0"),
    "sym20": ("symmetric", "0.2"),
    "sym40": ("symmetric", "0.4"),
    "sym60": ("symmetric", "0.6"),
    "sym80": ("symmetric", "0.8"),
    "asym20": ("asymmetric", "0.2"),
    "asym40": ("asymmetric", "0.4"),
    "dep20": ("dependent", "0.2"),
    "dep40": ("dependent", "0.4"),
    "dep60": ("dependent", "0.6"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "all"), default="all"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 124, 125))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=tuple(SCENARIOS),
    )
    parser.add_argument("--include-cifar-n", action="store_true")
    parser.add_argument("--cifar10n-labels", type=Path)
    parser.add_argument("--cifar100n-labels", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    datasets = ("cifar10", "cifar100") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        for seed in args.seeds:
            for scenario in args.scenarios:
                noise_type, noise_rate = SCENARIOS[scenario]
                command = [
                    sys.executable, str(repo / "main.py"),
                    "--dataset", dataset,
                    "--root", str(args.data_root),
                    "--noise-type", noise_type,
                    "--noise-rate", noise_rate,
                    "--noise-method", "method2",
                    "--seed", str(seed),
                    "--gpu", args.gpu,
                    "--workers", str(args.workers),
                    "--tag", f"sadt_{dataset}_{scenario}_s{seed}",
                    "--output-root", str(args.output_root),
                ]
                subprocess.run(command, cwd=repo, check=True)

            if args.include_cifar_n:
                if dataset == "cifar10":
                    label_path = args.cifar10n_labels
                    human_cases = ("worst", "aggre", "rand1", "rand2", "rand3")
                else:
                    label_path = args.cifar100n_labels
                    human_cases = ("noisy100",)
                if label_path is None:
                    raise ValueError(f"CIFAR-N label path is required for {dataset}")
                for human_case in human_cases:
                    command = [
                        sys.executable, str(repo / "main.py"),
                        "--dataset", dataset,
                        "--root", str(args.data_root),
                        "--noise-type", "human",
                        "--noise-rate", human_case,
                        "--human-label-path", str(label_path),
                        "--seed", str(seed),
                        "--gpu", args.gpu,
                        "--workers", str(args.workers),
                        "--tag", f"sadt_{dataset}_{human_case}_s{seed}",
                        "--output-root", str(args.output_root),
                    ]
                    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
