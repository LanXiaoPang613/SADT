#!/usr/bin/env python3
"""Compare three methods through bin-wise median-gradient contrasts.

This CPU-only post-processing script reads sample_gradients.csv and
metadata.json from one or more diagnostic directories.  It applies shared,
fixed checkpoint-difficulty bins, computes each method's median separately,
and then reports both SADT active and Full SADT relative to NCE+NNCE as:

    ratio       = median(method) / median(NCE+NNCE)
    log2-ratio  = log2(ratio)
    difference  = median(method) - median(NCE+NNCE)

The comparison CSV always contains all three measures.  --comparison only
selects which one is plotted.  Missing or empty bins remain explicit rows in
the CSV and appear as gaps; undefined zero-denominator/log2 cases are never
silently regularized with an arbitrary epsilon.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependent
    raise SystemExit(
        "plot_sadt_gradient_ratio_difference.py requires numpy and matplotlib."
    ) from exc


SCRIPT_VERSION = "2.0"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_THRESHOLD_LABEL_ROTATION = 0
SADT_METHOD = "sadt_active"
BASELINE_METHOD = "nce_nnce"
FULL_METHOD = "full_sadt"
TARGET_METHODS = (SADT_METHOD, BASELINE_METHOD, FULL_METHOD)
COMPARISON_METHODS = (SADT_METHOD, FULL_METHOD)
METHOD_ALIASES = {
    SADT_METHOD: SADT_METHOD,
    BASELINE_METHOD: BASELINE_METHOD,
    FULL_METHOD: FULL_METHOD,
    "full_sacjdtl": FULL_METHOD,
}
METHOD_LABELS = {
    SADT_METHOD: "SADT active (branch)",
    BASELINE_METHOD: "NCE+NNCE",
    FULL_METHOD: "Full SADT (complete loss)",
}
METHOD_STYLES = {
    SADT_METHOD: {
        "color": "#D55E00",
        "marker": "o",
        "linestyle": "-",
        "x_offset": -0.055,
    },
    FULL_METHOD: {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "-",
        "x_offset": 0.055,
    },
}
DEFAULT_INPUTS = (
    ("C10N", SCRIPT_DIR / "inputs" / "c10n_worst_s123_3060"),
    ("C100N", SCRIPT_DIR / "inputs" / "c100n_noisy_s123_jinge"),
)
DEFAULT_EDGES: Mapping[str, Tuple[float, ...]] = {
    "nll": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, math.inf),
    "p_y": (
        0.0,
        1e-32,
        1e-16,
        1e-8,
        1e-4,
        1e-2,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
    ),
}
BIN_COLUMNS = {"nll": "candidate_nll", "p_y": "candidate_probability"}
METRICS = (
    (
        "grad_candidate_abs",
        r"Candidate label $\left|\frac{\partial L}{\partial z_y}\right|$",
    ),
    (
        "gradient_l2",
        r"Full logit $\left\|\nabla_{\mathbf{z}} L\right\|_2$",
    ),
)
GROUP_LABELS = {"difficult_clean": "Difficult clean", "noisy": "Noisy"}
DATASET_CLASS_COUNTS = {"C10N": 10, "C100N": 100}
DATASET_DISPLAY_LABELS = {
    "C10N": "CIFAR-10N with worst labels",
    "C100N": "CIFAR-100N with noisy labels",
}
REQUIRED_COLUMNS = {
    "sample_index",
    "group",
    "method",
    "candidate_probability",
    "candidate_nll",
    "grad_candidate_abs",
    "gradient_l2",
}


@dataclass
class DiagnosticDataset:
    label: str
    source_dir: Path
    metadata: Dict[str, object]
    rows: List[MutableMapping[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot bin-wise median SADT-active and Full-SADT gradients relative "
            "to NCE+NNCE. With no arguments, packaged C10N/C100N inputs and "
            "<script-dir>/comparison_outputs/<comparison> are used."
        )
    )
    parser.add_argument("--c10n-dir", type=Path, help="CIFAR-10N diagnostic directory.")
    parser.add_argument("--c100n-dir", type=Path, help="CIFAR-100N diagnostic directory.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Additional diagnostic input; repeat as needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory (default: "
            "<script-dir>/comparison_outputs/<comparison>)."
        ),
    )
    parser.add_argument(
        "--comparison",
        choices=("ratio", "log2-ratio", "difference"),
        default="log2-ratio",
        help=(
            "Quantity to plot (default: log2-ratio). Direction is always "
            "each SADT variant relative to NCE+NNCE."
        ),
    )
    parser.add_argument(
        "--bin-by",
        choices=tuple(DEFAULT_EDGES),
        default="nll",
        help="Fixed checkpoint quantity used for bins (default: nll).",
    )
    parser.add_argument(
        "--edges",
        help="Optional comma-separated fixed edges; inf, +inf and -inf are accepted.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help=(
            "Minimum observations required from each method for a plotted point "
            "(default: 1). All bins remain in the CSV."
        ),
    )
    parser.add_argument(
        "--flag-count",
        type=int,
        default=10,
        help="Draw points with min(method n) below this count as hollow (default: 10).",
    )
    parser.add_argument(
        "--formats",
        default="svg,tiff",
        help="Comma-separated formats from png,pdf,svg (default: png,pdf).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Raster DPI (default: 220).")
    return parser


def parse_input_argument(value: str) -> Tuple[Optional[str], Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label, raw_path = label.strip(), raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"invalid --input value: {value!r}")
        return label, Path(raw_path)
    value = value.strip()
    if not value:
        raise ValueError("--input cannot be empty")
    return None, Path(value)


def derive_dataset_label(metadata: Mapping[str, object], source_dir: Path) -> str:
    dataset = str(metadata.get("dataset", "")).strip().lower().replace("-", "")
    if dataset == "cifar10":
        return "C10N"
    if dataset == "cifar100":
        return "C100N"
    return str(metadata.get("dataset") or source_dir.name)


def _float_field(row: Mapping[str, str], name: str, source: Path, line_no: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line_no}: invalid {name!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}:{line_no}: {name} must be finite")
    return value


def load_diagnostic(label: Optional[str], source_dir: Path) -> DiagnosticDataset:
    source_dir = source_dir.expanduser().resolve()
    csv_path = source_dir / "sample_gradients.csv"
    metadata_path = source_dir / "metadata.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing input: {csv_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing input: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")

    rows: List[MutableMapping[str, object]] = []
    seen_keys = set()
    candidate_by_sample: Dict[Tuple[str, int], Tuple[float, float]] = {}
    methods_by_sample: Dict[Tuple[str, int], set] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            raw_method = str(raw.get("method", "")).strip()
            method = METHOD_ALIASES.get(raw_method)
            if method is None:
                continue
            try:
                sample_index = int(raw["sample_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_no}: invalid sample_index") from exc
            group = str(raw.get("group", "")).strip()
            if not group:
                raise ValueError(f"{csv_path}:{line_no}: group must be non-empty")

            candidate_probability = _float_field(
                raw, "candidate_probability", csv_path, line_no
            )
            candidate_nll = _float_field(raw, "candidate_nll", csv_path, line_no)
            grad_candidate_abs = _float_field(
                raw, "grad_candidate_abs", csv_path, line_no
            )
            gradient_l2 = _float_field(raw, "gradient_l2", csv_path, line_no)
            if not -1e-12 <= candidate_probability <= 1.0 + 1e-12:
                raise ValueError(
                    f"{csv_path}:{line_no}: candidate_probability outside [0, 1]"
                )
            if candidate_nll < -1e-12:
                raise ValueError(f"{csv_path}:{line_no}: candidate_nll is negative")
            if grad_candidate_abs < 0.0 or gradient_l2 < 0.0:
                raise ValueError(
                    f"{csv_path}:{line_no}: gradient magnitudes must be nonnegative"
                )

            unique_key = (group, method, sample_index)
            if unique_key in seen_keys:
                raise ValueError(f"{csv_path}:{line_no}: duplicate row key {unique_key}")
            seen_keys.add(unique_key)

            sample_key = (group, sample_index)
            fixed_values = (candidate_nll, candidate_probability)
            previous = candidate_by_sample.get(sample_key)
            if previous is not None and not (
                math.isclose(previous[0], fixed_values[0], rel_tol=1e-12, abs_tol=1e-15)
                and math.isclose(previous[1], fixed_values[1], rel_tol=1e-12, abs_tol=1e-15)
            ):
                raise ValueError(
                    f"{csv_path}:{line_no}: checkpoint NLL/p_y differs between "
                    f"methods for sample {sample_key}"
                )
            if previous is None:
                candidate_by_sample[sample_key] = fixed_values
            else:
                # Canonicalize accepted round-off differences so one matched sample
                # can never straddle a fixed bin edge across the three methods.
                candidate_nll, candidate_probability = previous
            methods_by_sample[sample_key].add(method)
            rows.append(
                {
                    "sample_index": sample_index,
                    "group": group,
                    "method": method,
                    "candidate_probability": min(max(candidate_probability, 0.0), 1.0),
                    "candidate_nll": max(candidate_nll, 0.0),
                    "grad_candidate_abs": grad_candidate_abs,
                    "gradient_l2": gradient_l2,
                }
            )

    if not rows:
        raise ValueError(f"{csv_path} contains no {TARGET_METHODS} rows")
    present_methods = {str(row["method"]) for row in rows}
    missing_methods = set(TARGET_METHODS).difference(present_methods)
    if missing_methods:
        raise ValueError(f"{csv_path} lacks comparison methods {sorted(missing_methods)}")
    for sample_key, methods in methods_by_sample.items():
        missing_for_sample = set(TARGET_METHODS).difference(methods)
        if missing_for_sample:
            raise ValueError(
                f"{csv_path}: sample {sample_key} lacks comparison methods "
                f"{sorted(missing_for_sample)}"
            )

    return DiagnosticDataset(
        label=label or derive_dataset_label(metadata, source_dir),
        source_dir=source_dir,
        metadata=metadata,
        rows=rows,
    )


def parse_edges(raw_edges: Optional[str], bin_by: str) -> Tuple[List[float], str]:
    if raw_edges is None:
        return list(DEFAULT_EDGES[bin_by]), "default"
    aliases = {
        "inf": math.inf,
        "+inf": math.inf,
        "infinity": math.inf,
        "+infinity": math.inf,
        "-inf": -math.inf,
        "-infinity": -math.inf,
    }
    edges: List[float] = []
    for token in raw_edges.split(","):
        token = token.strip().lower()
        if not token:
            raise ValueError("--edges contains an empty token")
        try:
            value = aliases[token] if token in aliases else float(token)
        except ValueError as exc:
            raise ValueError(f"invalid edge token: {token!r}") from exc
        if math.isnan(value):
            raise ValueError("NaN is not a valid bin edge")
        edges.append(value)
    if len(edges) < 2:
        raise ValueError("--edges must contain at least two values")
    if any(not left < right for left, right in zip(edges, edges[1:])):
        raise ValueError("--edges must be strictly increasing")
    return edges, "custom"


def assign_bins(
    datasets: Sequence[DiagnosticDataset], bin_by: str, edges: Sequence[float]
) -> None:
    column = BIN_COLUMNS[bin_by]
    values = np.asarray(
        [float(row[column]) for dataset in datasets for row in dataset.rows], dtype=float
    )
    if values.min() < edges[0] or values.max() > edges[-1]:
        raise ValueError(
            f"fixed edges [{edges[0]}, {edges[-1]}] do not cover observed {column} "
            f"range [{values.min()}, {values.max()}]"
        )

    edge_array = np.asarray(edges, dtype=float)
    number_of_bins = len(edges) - 1
    for dataset in datasets:
        dataset_values = np.asarray([float(row[column]) for row in dataset.rows], dtype=float)
        indices = np.searchsorted(edge_array, dataset_values, side="right") - 1
        if math.isfinite(edges[-1]):
            indices[dataset_values == edges[-1]] = number_of_bins - 1
        if np.any(indices < 0) or np.any(indices >= number_of_bins):
            bad_value = dataset_values[(indices < 0) | (indices >= number_of_bins)][0]
            raise ValueError(f"value {bad_value!r} was not assigned to a fixed bin")
        for row, bin_index in zip(dataset.rows, indices.tolist()):
            row["bin_index"] = int(bin_index)


def ordered_groups(dataset: DiagnosticDataset) -> List[str]:
    present = {str(row["group"]) for row in dataset.rows}
    preferred = [name for name in ("difficult_clean", "noisy") if name in present]
    return preferred + sorted(present.difference(preferred))


def format_edge(value: float, unicode_infinity: bool = False) -> str:
    if math.isinf(value):
        if unicode_infinity:
            return "−∞" if value < 0 else "∞"
        return "-inf" if value < 0 else "inf"
    if value == 0.0:
        return "0"
    return f"{value:g}"


def interval_labels(edges: Sequence[float]) -> List[str]:
    labels = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        closing = "]" if index == len(edges) - 2 and math.isfinite(upper) else ")"
        labels.append(f"[{format_edge(lower)}, {format_edge(upper)}{closing}")
    return labels


def short_bin_labels(edges: Sequence[float]) -> List[str]:
    labels = []
    for lower, upper in zip(edges, edges[1:]):
        if math.isinf(upper) and upper > 0:
            labels.append(f"{format_edge(lower, True)}+")
        elif math.isinf(lower) and lower < 0:
            labels.append(f"< {format_edge(upper, True)}")
        else:
            labels.append(
                f"{format_edge(lower, True)}–{format_edge(upper, True)}"
            )
    return labels


def dataset_class_count(dataset: DiagnosticDataset) -> int:
    if dataset.label in DATASET_CLASS_COUNTS:
        return DATASET_CLASS_COUNTS[dataset.label]
    metadata_name = str(dataset.metadata.get("dataset", "")).lower().replace("-", "")
    if metadata_name == "cifar10":
        return 10
    if metadata_name == "cifar100":
        return 100
    raise ValueError(f"cannot infer class count for dataset {dataset.label!r}")


def value_to_bin_coordinate(
    value: float, edges: Sequence[float]
) -> Optional[float]:
    if value < edges[0] or value > edges[-1]:
        return None
    number_of_bins = len(edges) - 1
    edge_array = np.asarray(edges, dtype=float)
    bin_index = int(np.searchsorted(edge_array, value, side="right") - 1)
    if math.isfinite(edges[-1]) and value == edges[-1]:
        bin_index = number_of_bins - 1
    if bin_index < 0 or bin_index >= number_of_bins:
        return None
    lower = float(edges[bin_index])
    upper = float(edges[bin_index + 1])
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return float(bin_index)
    fraction = min(max((value - lower) / (upper - lower), 0.0), 1.0)
    return float(bin_index) - 0.5 + fraction


def reference_thresholds(
    dataset: DiagnosticDataset, bin_by: str, edges: Sequence[float]
) -> List[Dict[str, object]]:
    class_count = dataset_class_count(dataset)
    if bin_by == "nll":
        c_value = math.log(class_count)
        specs = [
            ("c_C", r"$ln(C)$", c_value, "--", "ln(C)"),
            ("2c_C", r"$2ln(C)$", 2.0 * c_value, ":", "2 ln(C)"),
        ]
    else:
        specs = [
            ("1_over_C", r"$1/C$", 1.0 / class_count, "--", "1/C"),
            (
                "1_over_C_squared",
                r"$1/C^2$",
                1.0 / (class_count * class_count),
                ":",
                "1/C^2",
            ),
        ]
    return [
        {
            "key": key,
            "label": label,
            "value": value,
            "linestyle": linestyle,
            "definition": definition,
            "bin_coordinate": value_to_bin_coordinate(value, edges),
        }
        for key, label, value, linestyle, definition in specs
    ]


def median(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=float)
    return None if array.size == 0 else float(np.median(array))


def comparison_values(
    method_median: Optional[float],
    baseline_median: Optional[float],
    method: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    if method_median is None and baseline_median is None:
        return None, None, None, "empty_both"
    if method_median is None:
        return None, None, None, f"missing_{method}"
    if baseline_median is None:
        return None, None, None, "missing_nce_nnce"

    difference = method_median - baseline_median
    if baseline_median == 0.0:
        status = "both_zero" if method_median == 0.0 else "zero_nce_nnce"
        return None, None, difference, status

    ratio = method_median / baseline_median
    if ratio == 0.0:
        return ratio, None, difference, f"zero_{method}"
    return ratio, math.log2(ratio), difference, "ok"


def build_comparison(
    datasets: Sequence[DiagnosticDataset], bin_by: str, edges: Sequence[float]
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    labels = interval_labels(edges)
    for dataset in datasets:
        lookup: Dict[Tuple[str, str, int, str], List[float]] = defaultdict(list)
        for row in dataset.rows:
            for metric, _ in METRICS:
                lookup[
                    (str(row["group"]), str(row["method"]), int(row["bin_index"]), metric)
                ].append(float(row[metric]))

        for group in ordered_groups(dataset):
            for bin_index, bin_label in enumerate(labels):
                record: Dict[str, object] = {
                    "dataset": dataset.label,
                    "metadata_dataset": dataset.metadata.get("dataset", ""),
                    "noise_type": dataset.metadata.get("noise_type", ""),
                    "noise_rate": dataset.metadata.get("noise_rate", ""),
                    "group": group,
                    "binning_variable": BIN_COLUMNS[bin_by],
                    "bin_index": bin_index,
                    "bin_label": bin_label,
                    "bin_lower": format_edge(edges[bin_index]),
                    "bin_upper": format_edge(edges[bin_index + 1]),
                    "comparison_direction": (
                        f"{METHOD_LABELS[SADT_METHOD]} and "
                        f"{METHOD_LABELS[FULL_METHOD]} relative to "
                        f"{METHOD_LABELS[BASELINE_METHOD]}"
                    ),
                }
                for metric, _ in METRICS:
                    baseline_values = lookup[(group, BASELINE_METHOD, bin_index, metric)]
                    baseline_median = median(baseline_values)
                    record.update(
                        {
                            f"{metric}_nce_nnce_n": len(baseline_values),
                            f"{metric}_nce_nnce_median": baseline_median,
                        }
                    )
                    for method in COMPARISON_METHODS:
                        method_values = lookup[(group, method, bin_index, metric)]
                        method_median = median(method_values)
                        ratio, log2_ratio, difference, status = comparison_values(
                            method_median, baseline_median, method
                        )
                        if (
                            method_values
                            and baseline_values
                            and len(method_values) != len(baseline_values)
                        ):
                            status += ";count_mismatch"
                        record.update(
                            {
                                f"{metric}_{method}_n": len(method_values),
                                f"{metric}_{method}_median": method_median,
                                f"{metric}_{method}_n_min": min(
                                    len(method_values), len(baseline_values)
                                ),
                                f"{metric}_{method}_ratio": ratio,
                                f"{metric}_{method}_log2_ratio": log2_ratio,
                                f"{metric}_{method}_difference": difference,
                                f"{metric}_{method}_status": status,
                            }
                        )
                records.append(record)
    return records


def write_comparison_csv(
    records: Sequence[Mapping[str, object]], output_path: Path
) -> None:
    identity_fields = [
        "dataset",
        "metadata_dataset",
        "noise_type",
        "noise_rate",
        "group",
        "binning_variable",
        "bin_index",
        "bin_label",
        "bin_lower",
        "bin_upper",
        "comparison_direction",
    ]
    metric_fields: List[str] = []
    for metric, _ in METRICS:
        metric_fields.extend(
            [f"{metric}_nce_nnce_n", f"{metric}_nce_nnce_median"]
        )
        for method in COMPARISON_METHODS:
            metric_fields.extend(
                [
                    f"{metric}_{method}_n",
                    f"{metric}_{method}_median",
                    f"{metric}_{method}_n_min",
                    f"{metric}_{method}_ratio",
                    f"{metric}_{method}_log2_ratio",
                    f"{metric}_{method}_difference",
                    f"{metric}_{method}_status",
                ]
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=identity_fields + metric_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def comparison_field(comparison: str) -> str:
    return comparison.replace("-", "_")


def comparison_axis_label(comparison: str) -> str:
    if comparison == "ratio":
        return (
            r"Gradient response "
            r"$\frac{\mathrm{median}(g_{\mathrm{method}})}"
            r"{\mathrm{median}(g_{\mathrm{NCE{+}NNCE}})}$"
        )
    if comparison == "log2-ratio":
        return (
            r"Gradient response "
            r"$\log_2\!\left(\frac{\mathrm{median}(g_{\mathrm{method}})}"
            r"{\mathrm{median}(g_{\mathrm{NCE{+}NNCE}})}\right)$"
        )
    return (
        r"Gradient response "
        r"$\mathrm{median}(g_{\mathrm{method}})-"
        r"\mathrm{median}(g_{\mathrm{NCE{+}NNCE}})$"
    )


def comparison_title(comparison: str) -> str:
    if comparison == "ratio":
        return "Median-ratio"
    if comparison == "log2-ratio":
        return r"Median $\log_2$ ratio"
    return "Median-difference"


def dataset_display_label(dataset: DiagnosticDataset) -> str:
    return DATASET_DISPLAY_LABELS.get(dataset.label, dataset.label)


def reference_value(comparison: str) -> float:
    return 1.0 if comparison == "ratio" else 0.0


def records_for_panel(
    records: Sequence[Mapping[str, object]], dataset: str, group: str
) -> Dict[int, Mapping[str, object]]:
    return {
        int(record["bin_index"]): record
        for record in records
        if record["dataset"] == dataset and record["group"] == group
    }


def metric_limits(
    records: Sequence[Mapping[str, object]],
    dataset_label: str,
    comparison: str,
    min_count: int,
) -> Dict[str, Tuple[float, float]]:
    suffix = comparison_field(comparison)
    limits: Dict[str, Tuple[float, float]] = {}
    for metric, _ in METRICS:
        values = [
            float(record[f"{metric}_{method}_{suffix}"])
            for record in records
            for method in COMPARISON_METHODS
            if record["dataset"] == dataset_label
            and record[f"{metric}_{method}_{suffix}"] is not None
            and int(record[f"{metric}_{method}_n_min"]) >= min_count
            and math.isfinite(float(record[f"{metric}_{method}_{suffix}"]))
        ]
        ref = reference_value(comparison)
        observed = values + [ref]
        lower = min(observed)
        upper = max(observed)
        if comparison == "ratio":
            padding = max(0.10 * (upper - lower), 0.06)
            limits[metric] = (max(0.0, lower - padding), upper + padding)
        else:
            span = upper - lower
            if span > 0.0:
                padding = 0.10 * span
            else:
                padding = 0.25 if comparison == "log2-ratio" else 1e-6
            limits[metric] = (lower - padding, upper + padding)
    return limits


def safe_stem(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._-")
    return value or "dataset"


def dataset_figure_paths(
    output_dir: Path,
    dataset_label: str,
    comparison: str,
    formats: Sequence[str],
) -> List[Path]:
    stem = output_dir / (
        f"{safe_stem(dataset_label)}_gradient_median_"
        f"{comparison.replace('-', '_')}"
    )
    return [Path(f"{stem}.{file_format}") for file_format in formats]


def validate_unique_output_paths(paths: Sequence[Path]) -> None:
    seen: Dict[str, Path] = {}
    for path in paths:
        key = str(path.resolve()).casefold()
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"output path collision after filename normalization: "
                f"{previous} and {path}"
            )
        seen[key] = path


def plot_dataset(
    dataset: DiagnosticDataset,
    records: Sequence[Mapping[str, object]],
    edges: Sequence[float],
    bin_by: str,
    comparison: str,
    limits: Mapping[str, Tuple[float, float]],
    min_count: int,
    flag_count: int,
    output_paths: Sequence[Path],
    dpi: int,
) -> List[Path]:
    groups = ordered_groups(dataset)
    number_of_bins = len(edges) - 1
    x = np.arange(number_of_bins)
    labels = short_bin_labels(edges)
    suffix = comparison_field(comparison)
    ref = reference_value(comparison)
    threshold_specs = reference_thresholds(dataset, bin_by, edges)
    x_axis_label = (
        r"Candidate label bins with $-\log p_y$"
        if bin_by == "nll"
        else r"Checkpoint candidate-label probability $p_y$ bin"
    )

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(rc):
        figure, axes = plt.subplots(
            len(groups),
            len(METRICS),
            figsize=(11.0, 8.5),
            squeeze=False,
            sharex="col",
            sharey="col",
        )
        for row_index, group in enumerate(groups):
            panel = records_for_panel(records, dataset.label, group)
            for column_index, (metric, metric_label) in enumerate(METRICS):
                axis = axes[row_index, column_index]
                any_finite = False
                for method in COMPARISON_METHODS:
                    style = METHOD_STYLES[method]
                    method_x = x + float(style["x_offset"])
                    values = np.full(number_of_bins, np.nan)
                    counts = np.zeros(number_of_bins, dtype=int)
                    for bin_index in range(number_of_bins):
                        record = panel.get(bin_index)
                        if record is None:
                            continue
                        counts[bin_index] = int(
                            record[f"{metric}_{method}_n_min"]
                        )
                        value = record[f"{metric}_{method}_{suffix}"]
                        if (
                            value is not None
                            and counts[bin_index] >= min_count
                            and math.isfinite(float(value))
                        ):
                            values[bin_index] = float(value)

                    finite = np.isfinite(values)
                    any_finite = any_finite or bool(np.any(finite))
                    axis.plot(
                        method_x,
                        values,
                        color=str(style["color"]),
                        linestyle=str(style["linestyle"]),
                        linewidth=1.9,
                        zorder=3,
                    )
                    if not np.any(finite):
                        continue
                    normal = finite.copy()
                    normal[finite] = counts[finite] >= flag_count
                    low_count = finite & ~normal
                    if np.any(normal):
                        axis.scatter(
                            method_x[normal],
                            values[normal],
                            s=23,
                            marker=str(style["marker"]),
                            facecolors=str(style["color"]),
                            edgecolors="white",
                            linewidths=0.55,
                            zorder=4,
                        )
                    if np.any(low_count):
                        axis.scatter(
                            method_x[low_count],
                            values[low_count],
                            s=27,
                            marker=str(style["marker"]),
                            facecolors="white",
                            edgecolors=str(style["color"]),
                            linewidths=1.15,
                            zorder=5,
                        )
                if not any_finite:
                    axis.text(
                        0.5,
                        0.5,
                        "No finite bins at this count threshold",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="#666666",
                    )

                for threshold in threshold_specs:
                    coordinate = threshold["bin_coordinate"]
                    if coordinate is None:
                        continue
                    axis.axvline(
                        float(coordinate),
                        color="#5F5F5F",
                        linestyle=str(threshold["linestyle"]),
                        linewidth=0.9,
                        alpha=0.58,
                        zorder=1,
                    )
                    axis.text(
                        float(coordinate) + 0.035,
                        0.035,
                        threshold["label"],
                        transform=axis.get_xaxis_transform(),
                        rotation=DEFAULT_THRESHOLD_LABEL_ROTATION,
                        ha="left",
                        va="bottom",
                        fontsize=14,
                        color="#525252",
                    )

                axis.axhline(
                    ref,
                    color="#737373",
                    linewidth=0.7,
                    alpha=0.75,
                    zorder=1,
                )
                axis.set_ylim(*limits[metric])
                axis.set_xlim(-0.35, number_of_bins - 0.65)
                axis.grid(
                    axis="y",
                    which="major",
                    color="#A9A9A9",
                    alpha=0.22,
                    linewidth=0.55,
                )
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["left"].set_color("#888888")
                axis.spines["bottom"].set_color("#888888")
                axis.set_xticks(x)
                if row_index == len(groups) - 1:
                    axis.set_xticklabels(labels, rotation=30, ha="right")
                else:
                    axis.tick_params(axis="x", labelbottom=False)
                if row_index == 0:
                    axis.set_title(metric_label, pad=8)
                axis.text(
                    0.90,
                    0.96,
                    GROUP_LABELS.get(group, group),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=14,
                )

        hollow_note = (
            f"hollow marker: min(method n) < {flag_count}"
            if flag_count > 0
            else ""
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=str(METHOD_STYLES[SADT_METHOD]["marker"]),
                color=str(METHOD_STYLES[SADT_METHOD]["color"]),
                markerfacecolor=str(METHOD_STYLES[SADT_METHOD]["color"]),
                markeredgecolor="white",
                linestyle=str(METHOD_STYLES[SADT_METHOD]["linestyle"]),
                linewidth=1.9,
                markersize=5,
                label=METHOD_LABELS[SADT_METHOD],
            ),
            Line2D(
                [0],
                [0],
                marker=str(METHOD_STYLES[FULL_METHOD]["marker"]),
                color=str(METHOD_STYLES[FULL_METHOD]["color"]),
                markerfacecolor=str(METHOD_STYLES[FULL_METHOD]["color"]),
                markeredgecolor="white",
                linestyle=str(METHOD_STYLES[FULL_METHOD]["linestyle"]),
                linewidth=1.9,
                markersize=5,
                label=METHOD_LABELS[FULL_METHOD],
            ),
        ]
        if flag_count > 0:
            legend_handles.append(
                Line2D(
                    [0], [0], marker="o", color="none", markerfacecolor="white",
                    markeredgecolor="#555555", markersize=5, label=hollow_note
                )
            )
        # figure.suptitle(
        #     f"{comparison_title(comparison)} gradient response under "
        #     f"{dataset_display_label(dataset)}",
        #     fontsize=16,
        #     y=0.998,
        # )
        # figure.legend(
        #     handles=legend_handles,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 0.982),
        #     ncol=len(legend_handles),
        #     frameon=False,
        #     columnspacing=1.8,
        # )
        figure.suptitle(
            f"{comparison_title(comparison)} gradient response under "
            f"{dataset_display_label(dataset)}",
            fontsize=16,
            y=0.99,
        )

        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.8,
        )
        # figure.supxlabel(x_axis_label, y=0.082, fontsize=16)
        # figure.supylabel(comparison_axis_label(comparison), x=0.012, fontsize=9.5)
        figure.supxlabel(
            x_axis_label,
            x=0.5,
            y=0.035,
            fontsize=16,
        )

        figure.supylabel(
            comparison_axis_label(comparison),
            x=0.025,
            y=0.5,
            fontsize=16,
        )
        reference_text = "1" if comparison == "ratio" else "0"
        class_count = dataset_class_count(dataset)
        if bin_by == "nll":
            c_value = math.log(class_count)
            vertical_reference_note = (
                rf"Vertical references: $c_C=\ln({class_count})={c_value:.6g}$ "
                rf"(dashed); $2c_C={2.0 * c_value:.6g}$ (dotted)."
            )
        else:
            vertical_reference_note = (
                rf"Vertical references: $1/C={1.0 / class_count:.6g}$ (dashed); "
                rf"$1/C^2={1.0 / (class_count * class_count):.6g}$ (dotted)."
            )
        # figure.text(
        #     0.5,
        #     0.012,
        #     "Fixed-checkpoint instantaneous logit response—not cumulative "
        #     "parameter updates.\n"
        #     f"Both SADT series are relative to NCE+NNCE; reference = "
        #     f"{reference_text}. Gaps are empty, undefined, or below "
        #     f"--min-count={min_count} bins.\n{vertical_reference_note}",
        #     ha="center",
        #     va="bottom",
        #     fontsize=7.8,
        #     color="#4A4A4A",
        # )
        # figure.subplots_adjust(
        #     left=0.095,
        #     right=0.985,
        #     bottom=0.205,
        #     top=0.895,
        #     hspace=0.23,
        #     wspace=0.16,
        # )
        figure.subplots_adjust(
            left=0.095,
            right=0.985,
            bottom=0.16,
            top=0.80,
            hspace=0.30,
            wspace=0.16,
        )

        written: List[Path] = []
        for output_path in output_paths:
            file_format = output_path.suffix.lstrip(".").lower()
            options = {"bbox_inches": "tight"}
            if file_format == "tiff":
                options["dpi"] = dpi
            figure.savefig(output_path, **options)
            written.append(output_path)
        plt.close(figure)
    return written


def parse_formats(raw: str) -> List[str]:
    allowed = {"png", "tiff", "svg"}
    formats: List[str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"unsupported format {token!r}; choose from {sorted(allowed)}")
        if token not in formats:
            formats.append(token)
    if not formats:
        raise ValueError("--formats must contain at least one supported format")
    return formats


def write_metadata(
    output_path: Path,
    datasets: Sequence[DiagnosticDataset],
    edges: Sequence[float],
    edge_source: str,
    bin_by: str,
    comparison: str,
    min_count: int,
    flag_count: int,
    formats: Sequence[str],
    dpi: int,
    outputs: Sequence[Path],
) -> None:
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "name": script_path.name,
            "version": SCRIPT_VERSION,
            "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "execution": "CPU-only post-processing; no model forward/backward pass",
        "comparison": {
            "plotted": comparison,
            "methods": [
                METHOD_LABELS[SADT_METHOD],
                METHOD_LABELS[BASELINE_METHOD],
                METHOD_LABELS[FULL_METHOD],
            ],
            "direction": (
                "SADT active branch and complete Full SADT, each relative "
                "to NCE+NNCE"
            ),
            "method_roles": {
                METHOD_LABELS[SADT_METHOD]: "active branch only",
                METHOD_LABELS[BASELINE_METHOD]: "comparison baseline",
                METHOD_LABELS[FULL_METHOD]: "complete loss",
            },
            "csv_measures": ["ratio", "log2_ratio", "difference"],
            "definitions": {
                "ratio": "median(method) / median(NCE+NNCE)",
                "log2_ratio": "log2(ratio)",
                "difference": "median(method) - median(NCE+NNCE)",
            },
            "zero_policy": (
                "No pseudocount. Undefined division/log2 values are blank and "
                "described by each metric_status column."
            ),
            "estimator_note": (
                "These are ratios/differences of the two within-bin medians, not "
                "medians of paired per-sample ratios/differences."
            ),
            "interpretation": (
                "Fixed-checkpoint instantaneous logit-gradient response; not a "
                "cumulative parameter update over training."
            ),
            "sample_alignment_validation": (
                "Every (group, sample_index) is required to contain all three "
                "canonical methods and identical checkpoint candidate "
                "NLL/probability values."
            ),
            "min_count": min_count,
            "hollow_marker_below_n": flag_count,
        },
        "binning": {
            "selector": bin_by,
            "source_column": BIN_COLUMNS[bin_by],
            "edge_source": edge_source,
            "edges": [format_edge(value) if math.isinf(value) else value for value in edges],
            "interval_rule": (
                "left-closed/right-open [lower, upper), except a finite final upper "
                "edge is included"
            ),
            "shared_across_datasets_and_methods": True,
        },
        "statistics": {
            "center": "NumPy median",
            "quantile_method": "not applicable; this script does not draw uncertainty bands",
        },
        "figure": {
            "formats": list(formats),
            "png_dpi": dpi,
            "reference_lines": {
                dataset.label: {
                    "class_count": dataset_class_count(dataset),
                    "coordinate_rule": (
                        "linear interpolation between the left and right "
                        "categorical boundaries of the containing numeric bin; "
                        "an unbounded bin uses its center"
                    ),
                    "thresholds": [
                        {
                            "key": threshold["key"],
                            "definition": threshold["definition"],
                            "value": threshold["value"],
                            "linestyle": (
                                "dashed"
                                if threshold["linestyle"] == "--"
                                else "dotted"
                            ),
                            "bin_coordinate": threshold["bin_coordinate"],
                        }
                        for threshold in reference_thresholds(
                            dataset, bin_by, edges
                        )
                    ],
                }
                for dataset in datasets
            },
        },
        "inputs": [
            {
                "label": dataset.label,
                "directory": str(dataset.source_dir),
                "comparison_rows": len(dataset.rows),
                "sample_gradients_sha256": hashlib.sha256(
                    (dataset.source_dir / "sample_gradients.csv").read_bytes()
                ).hexdigest(),
                "checkpoint_sha256": (
                    dataset.metadata.get("checkpoint", {}).get("sha256")
                    if isinstance(dataset.metadata.get("checkpoint"), dict)
                    else None
                ),
            }
            for dataset in datasets
        ],
        "outputs": [path.name for path in outputs],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.min_count <= 0:
        raise SystemExit("--min-count must be positive")
    if args.flag_count < 0:
        raise SystemExit("--flag-count must be nonnegative")

    input_specs: List[Tuple[Optional[str], Path]] = []
    if args.c10n_dir is not None:
        input_specs.append(("C10N", args.c10n_dir))
    if args.c100n_dir is not None:
        input_specs.append(("C100N", args.c100n_dir))
    try:
        input_specs.extend(parse_input_argument(value) for value in args.input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not input_specs:
        input_specs.extend(DEFAULT_INPUTS)

    try:
        datasets = [load_diagnostic(label, path) for label, path in input_specs]
        labels = [dataset.label for dataset in datasets]
        if len(labels) != len(set(labels)):
            raise ValueError("dataset labels must be unique")
        edges, edge_source = parse_edges(args.edges, args.bin_by)
        formats = parse_formats(args.formats)
        assign_bins(datasets, args.bin_by, edges)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"input validation failed: {exc}") from exc

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else SCRIPT_DIR / "comparison_outputs" / comparison_field(args.comparison)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_comparison(datasets, args.bin_by, edges)
    csv_path = output_dir / "gradient_median_comparison.csv"
    metadata_path = output_dir / "gradient_median_comparison_metadata.json"
    planned_figures = {
        dataset.label: dataset_figure_paths(
            output_dir, dataset.label, args.comparison, formats
        )
        for dataset in datasets
    }
    figure_paths = [
        path
        for dataset in datasets
        for path in planned_figures[dataset.label]
    ]
    try:
        validate_unique_output_paths([csv_path, metadata_path] + figure_paths)
    except ValueError as exc:
        raise SystemExit(f"output validation failed: {exc}") from exc

    write_comparison_csv(records, csv_path)

    for dataset in datasets:
        limits = metric_limits(
            records, dataset.label, args.comparison, args.min_count
        )
        written = plot_dataset(
            dataset,
            records,
            edges,
            args.bin_by,
            args.comparison,
            limits,
            args.min_count,
            args.flag_count,
            planned_figures[dataset.label],
            args.dpi,
        )
        if written != planned_figures[dataset.label]:
            raise RuntimeError("internal error: written figure paths changed")

    all_outputs = [csv_path, metadata_path] + figure_paths
    write_metadata(
        metadata_path,
        datasets,
        edges,
        edge_source,
        args.bin_by,
        args.comparison,
        args.min_count,
        args.flag_count,
        formats,
        args.dpi,
        all_outputs,
    )

    finite_counts: Dict[str, Dict[str, int]] = {}
    suffix = comparison_field(args.comparison)
    for method in COMPARISON_METHODS:
        finite_counts[METHOD_LABELS[method]] = {}
        for metric, _ in METRICS:
            finite_counts[METHOD_LABELS[method]][metric] = sum(
                record[f"{metric}_{method}_{suffix}"] is not None
                and int(record[f"{metric}_{method}_n_min"]) >= args.min_count
                and math.isfinite(float(record[f"{metric}_{method}_{suffix}"]))
                for record in records
            )
    print(f"Wrote {len(records)} bin rows to {csv_path}")
    print(f"Finite plotted bins by metric: {finite_counts}")
    print(f"Recorded definitions and zero/count policy in {metadata_path}")
    for path in figure_paths:
        print(f"Wrote figure {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
