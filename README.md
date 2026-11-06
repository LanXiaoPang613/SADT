# Surprise annealed dual tail loss for robust noisy labels learning

This directory is the source release for
**Surprise-Annealed Dual-Tail Loss (SADT)** on CIFAR-10 and CIFAR-100. 

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

## Data layout

Raw data is deliberately excluded. A convenient shared root is:

```text
DATA_ROOT/
├── cifar10/
│   └── cifar-10-batches-py/
└── cifar100/
    └── cifar-100-python/
```

The package covers:

| Dataset | Clean | Symmetric | Asymmetric | Dependent/IDN | CIFAR-N |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | yes | yes | yes | 0.2/0.4/0.6 | clean, worst, aggre, rand1-3 |
| CIFAR-100 | yes | yes | yes | 0.2/0.4/0.6 | clean100, noisy100 |

Synthetic noise uses `method2` by default, matching the released paper runs. The six dependent-noise
index/configuration JSON files are bundled under
`datasets/data_dependent/config`. Random generation is seeded before loader
construction, so each seed recreates its corresponding label index.

For CIFAR-N, download the official label files and either place them at:

```text
datasets/data_human/CIFAR-10_human.pt
datasets/data_human/CIFAR-100_human.pt
```

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

