"""Paper settings for the CIFAR SADT experiments."""

from dataclasses import dataclass, replace

from sadt import SADTLoss


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str
    epochs: int
    learning_rate: float
    batch_size: int
    alpha: float
    beta: float
    l1_weight_decay: float
    l2_weight_decay: float
    architecture: str


PAPER_CONFIGS = {
    "cifar10": TrainingConfig(
        dataset="cifar10",
        epochs=120,
        learning_rate=0.01,
        batch_size=128,
        alpha=5.0,
        beta=1.0,
        l1_weight_decay=5e-5,
        l2_weight_decay=0.0,
        architecture="cnn8",
    ),
    "cifar100": TrainingConfig(
        dataset="cifar100",
        epochs=200,
        learning_rate=0.1,
        batch_size=128,
        alpha=10.0,
        beta=1.0,
        l1_weight_decay=5e-7,
        l2_weight_decay=0.0,
        architecture="resnet34",
    ),
}


def resolve_training_config(
    dataset: str,
    *,
    epochs: int | None = None,
    learning_rate: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> TrainingConfig:
    config = PAPER_CONFIGS[dataset]
    return replace(
        config,
        epochs=config.epochs if epochs is None else epochs,
        learning_rate=(
            config.learning_rate if learning_rate is None else learning_rate
        ),
        alpha=config.alpha if alpha is None else alpha,
        beta=config.beta if beta is None else beta,
    )


def build_loss(config: TrainingConfig) -> SADTLoss:
    return SADTLoss(alpha=config.alpha, beta=config.beta)
