# SADT: clean CIFAR reproduction package

This directory is the minimal, self-contained source release for
**Surprise-Annealed Dual-Tail Loss (SADT)** on CIFAR-10 and CIFAR-100. It was
distilled from the 2026-10-19 public SADT backup. It contains no datasets,
logs, checkpoints, TensorBoard events, host-specific launchers, or superseded
loss functions.

The proposed method is named `SADT` everywhere. The implementation is
[`sadt/loss.py`](sadt/loss.py), and the paper defaults are in
[`config.py`](config.py).

## Environment

The reported experiments use Python 3.10.13, PyTorch 2.1.1+cu118 and
torchvision 0.16.1+cu118.

```bash
conda env create -f environment.yml
conda activate sadt
```

or:

```bash
python -m pip install -r requirements.txt
```

Select a PyTorch/CUDA build compatible with the target driver if CUDA 11.8 is
not available.

## Data layout

Raw data is deliberately excluded. A convenient shared root is:

```text
DATA_ROOT/
├── cifar10/
│   └── cifar-10-batches-py/
└── cifar100/
    └── cifar-100-python/
```

The loaders can download CIFAR-10/100 through torchvision when files are
missing. `--root` may also point directly to a directory that contains
`cifar-10-batches-py` or `cifar-100-python`.

The package covers:

| Dataset | Clean | Symmetric | Asymmetric | Dependent/IDN | CIFAR-N |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | yes | yes | yes | 0.2/0.4/0.6 | clean, worst, aggre, rand1-3 |
| CIFAR-100 | yes | yes | yes | 0.2/0.4/0.6 | clean100, noisy100 |

Synthetic noise uses `method2` by default, matching the released paper runs;
`method1` is retained as a compatibility option. The six dependent-noise
index/configuration JSON files are bundled under
`datasets/data_dependent/config`. Random generation is seeded before loader
construction, so each seed recreates its corresponding label index.

For CIFAR-N, download the official label files and either place them at:

```text
datasets/data_human/CIFAR-10_human.pt
datasets/data_human/CIFAR-100_human.pt
```

or pass an absolute path with `--human-label-path`. Label-key mappings are
implemented in `datasets/data_loaders.py`.

## Paper defaults

| Setting | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Network | 8-layer CNN | ResNet-34 |
| Epochs | 120 | 200 |
| Initial LR | 0.01 | 0.1 |
| Batch size | 128 | 128 |
| Optimizer | SGD, momentum 0.9 | SGD, momentum 0.9 |
| Scheduler | cosine, `T_max=epochs` | cosine, `T_max=epochs` |
| Gradient clip | 5 | 5 |
| SADT `alpha` | 5 | 10 |
| SADT `beta` | 1 | 1 |
| L1 weight decay | 5e-5 | 5e-7 |

## Single-run commands

Clean labels are represented by symmetric noise with rate zero:

```bash
python main.py --dataset cifar10 --root DATA_ROOT --noise-type symmetric --noise-rate 0.0 --seed 123 --gpu 0 --tag c10_clean_s123
python main.py --dataset cifar100 --root DATA_ROOT --noise-type symmetric --noise-rate 0.0 --seed 123 --gpu 0 --tag c100_clean_s123
```

Symmetric noise:

```bash
python main.py --dataset cifar10 --root DATA_ROOT --noise-type symmetric --noise-rate 0.8 --seed 123 --gpu 0 --tag c10_sym08_s123
python main.py --dataset cifar100 --root DATA_ROOT --noise-type symmetric --noise-rate 0.8 --seed 123 --gpu 0 --tag c100_sym08_s123
```

Asymmetric noise:

```bash
python main.py --dataset cifar10 --root DATA_ROOT --noise-type asymmetric --noise-rate 0.4 --seed 123 --gpu 0 --tag c10_asym04_s123
python main.py --dataset cifar100 --root DATA_ROOT --noise-type asymmetric --noise-rate 0.4 --seed 123 --gpu 0 --tag c100_asym04_s123
```

Dependent/instance-dependent noise:

```bash
python main.py --dataset cifar10 --root DATA_ROOT --noise-type dependent --noise-rate 0.4 --seed 123 --gpu 0 --tag c10_dep04_s123
python main.py --dataset cifar100 --root DATA_ROOT --noise-type dependent --noise-rate 0.4 --seed 123 --gpu 0 --tag c100_dep04_s123
```

CIFAR-N:

```bash
python main.py --dataset cifar10 --root DATA_ROOT --noise-type human --noise-rate worst --human-label-path /path/CIFAR-10_human.pt --seed 123 --gpu 0 --tag c10n_worst_s123
python main.py --dataset cifar100 --root DATA_ROOT --noise-type human --noise-rate noisy100 --human-label-path /path/CIFAR-100_human.pt --seed 123 --gpu 0 --tag c100n_noisy_s123
```

Use `--dry-run` to resolve and print a configuration without loading data.

## Three-seed matrix

```bash
python scripts/run_cifar_formal.py --data-root DATA_ROOT --dataset all --gpu 0 --seeds 123 124 125
```

Add CIFAR-N with:

```bash
python scripts/run_cifar_formal.py --data-root DATA_ROOT --dataset all --gpu 0 --include-cifar-n --cifar10n-labels /path/CIFAR-10_human.pt --cifar100n-labels /path/CIFAR-100_human.pt
```

## Output structure

```text
results/<dataset>/SADT/<noise_type>_<noise_rate>/<tag>/
├── result.log
├── acc_seed<seed>.log
├── summary.json
├── events.out.tfevents.*          # when tensorboard is installed
└── model_seed<seed>_final.pth     # only with --save-final-model
```

## Verification

```bash
python -m compileall -q .
python -m unittest tests.test_sadt
python main.py --dataset cifar10 --root . --dry-run
python scripts/verify_package.py
```

`MANIFEST.sha256` records every packaged file except itself. See
`EXCLUDED.md` for the exclusion policy and `PACKAGE_PLAN.json` for source
provenance. The included `LICENSE` is an all-rights-reserved notice because
the source project did not specify an open-source license.
