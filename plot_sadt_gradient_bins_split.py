#!/usr/bin/env python3
"""Plot the SADT gradient-bin comparison as one 2x2 figure per dataset.

The script is a separate entry point from ``plot_sadt_gradient_bins.py``.  It
reuses that script's validated loading, fixed-bin assignment, and summary
functions so the statistical meaning stays identical:

* rows: difficult-clean and noisy samples;
* columns: candidate-label and full-logit gradient magnitudes;
* methods: SADT active branch, NCE+NNCE, and the complete Full SADT loss;
* curves/whiskers: median and interquartile range in fixed checkpoint bins;
* scale: symlog, with one shared range per metric across both rows of a figure.
* outputs: raw absolute response and method-specific shape-normalized response;
* markers: filled by default; hollow when the minimum bin count across methods
  is below the configured reliability threshold, as stated in a second legend.

The default IQR display uses per-bin whiskers, avoiding the triangular polygons
that a connected band can create next to an extreme or exact-zero bin.  Use
``--iqr-style band`` to recover the shaded-band presentation.

With the reproduction package layout, running the script without arguments is
enough to create raw and ``*_normalized`` C10N/C100N figures in
``split_outputs``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import plot_sadt_gradient_bins as base


SCRIPT_VERSION = "1.3"
PROJECT_DIR = Path(__file__).resolve().parent

# ======================== zero-argument user configuration ========================
# Edit only this block when moving the package or changing the default rendering.
# Running ``python plot_sadt_gradient_bins_split.py`` uses these values directly.
DEFAULT_C10N_DIR = PROJECT_DIR / "inputs" / "c10n_worst_s123_3060"
DEFAULT_C100N_DIR = PROJECT_DIR / "inputs" / "c100n_noisy_s123_jinge"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "split_outputs"
DEFAULT_BIN_BY = "nll"
DEFAULT_CUSTOM_EDGES: Optional[str] = None  # None -> fixed 0,1,2,...,256,inf bins
DEFAULT_FORMATS = "svg,tiff"
DEFAULT_IQR_STYLE = "whisker"  # whisker | band | none
DEFAULT_FLAG_COUNT = 10  # hollow when min_m n_m is below this threshold
DEFAULT_DPI = 600
DEFAULT_THRESHOLD_LABEL_ROTATION = 0  # 0 keeps c_C and 2c_C horizontal
# ================================================================================

SPLIT_METHODS = ("sadt_active", "nce_nnce", "full_sadt")
SPLIT_METHOD_ALIASES = {"full_sacjdtl": "full_sadt"}
SPLIT_METHOD_LABELS = {
    "sadt_active": "SADT active (branch)",
    "nce_nnce": "NCE+NNCE",
    "full_sadt": "Full SADT (complete loss)",
}
DATASET_CLASS_COUNTS = {"CIFAR-10N with worst labels": 10, "CIFAR-100N with noisy labels": 100}
SPLIT_REQUIRED_COLUMNS = {
    "sample_index",
    "group",
    "method",
    "candidate_probability",
    "candidate_nll",
    "grad_candidate_abs",
    "gradient_l2",
}
DATASET_INPUTS = {
    "CIFAR-10N with worst labels": DEFAULT_C10N_DIR,
    "CIFAR-100N with noisy labels": DEFAULT_C100N_DIR,
}
OUTPUT_STEMS = {
    "CIFAR-10N with worst labels": "gradient_bins_c10n",
    "CIFAR-100N with noisy labels": "gradient_bins_c100n",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create raw and shape-normalized C10N/C100N 2x2 median/IQR "
            "gradient figures using fixed checkpoint bins."
        )
    )
    parser.add_argument(
        "--c10n-dir",
        type=Path,
        default=DATASET_INPUTS["CIFAR-10N with worst labels"],
        help="CIFAR-10N diagnostic directory (defaults to the packaged input).",
    )
    parser.add_argument(
        "--c100n-dir",
        type=Path,
        default=DATASET_INPUTS["CIFAR-100N with noisy labels"],
        help="CIFAR-100N diagnostic directory (defaults to the packaged input).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: ./split_outputs beside this script).",
    )
    parser.add_argument(
        "--bin-by",
        choices=tuple(base.DEFAULT_EDGES),
        default=DEFAULT_BIN_BY,
        help="Fixed checkpoint quantity used for bins (default: nll).",
    )
    parser.add_argument(
        "--edges",
        default=DEFAULT_CUSTOM_EDGES,
        help="Optional comma-separated fixed edges; inf, +inf and -inf are accepted.",
    )
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="Comma-separated formats chosen from png,pdf,svg,tiff (default: svg,tiff).",
    )
    parser.add_argument(
        "--iqr-style",
        choices=("whisker", "band", "none"),
        default=DEFAULT_IQR_STYLE,
        help=(
            "IQR rendering: per-bin whiskers, shaded band, or hidden "
            "(default: whisker)."
        ),
    )
    parser.add_argument(
        "--flag-count",
        type=int,
        default=DEFAULT_FLAG_COUNT,
        help=(
            "Draw a non-empty bin's method markers hollow when the minimum count "
            "across methods is below this threshold; 0 disables the indicator "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help="Raster DPI (default: 600)."
    )
    return parser


def _float_field(
    row: Mapping[str, str], name: str, source: Path, line_no: int
) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line_no}: invalid {name!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}:{line_no}: {name} must be finite, got {value!r}")
    return value


def load_split_diagnostic(label: str, source_dir: Path) -> base.DiagnosticDataset:
    """Load a diagnostic while requiring the three split-figure methods.

    Keeping the method contract local makes this entry point immune to method
    additions in the original plotting script.  A legacy Full-SADT identifier is
    normalized internally; rows outside the explicit figure contract are ignored.
    """

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
        missing = SPLIT_REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            try:
                sample_index = int(raw["sample_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path}:{line_no}: invalid sample_index"
                ) from exc
            group = str(raw["group"]).strip()
            method = str(raw["method"]).strip()
            method = SPLIT_METHOD_ALIASES.get(method, method)
            if not group or not method:
                raise ValueError(
                    f"{csv_path}:{line_no}: group and method must be non-empty"
                )
            if method not in SPLIT_METHODS:
                continue

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
                and math.isclose(
                    previous[1], fixed_values[1], rel_tol=1e-12, abs_tol=1e-15
                )
            ):
                raise ValueError(
                    f"{csv_path}:{line_no}: checkpoint NLL/p_y differs across "
                    f"methods for sample {sample_key}"
                )
            candidate_by_sample[sample_key] = fixed_values
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
        raise ValueError(f"{csv_path} contains no data rows")
    for sample_key, methods in methods_by_sample.items():
        missing_methods = set(SPLIT_METHODS).difference(methods)
        if missing_methods:
            raise ValueError(
                f"{csv_path}: sample {sample_key} lacks split-figure methods "
                f"{sorted(missing_methods)}"
            )

    return base.DiagnosticDataset(
        label=label,
        source_dir=source_dir,
        metadata=metadata,
        rows=rows,
    )


def _short_number(value: float) -> str:
    if math.isinf(value):
        return "−∞" if value < 0 else "∞"
    if value == 0:
        return "0"
    absolute = abs(value)
    if absolute < 1e-3 or absolute >= 1e4:
        return f"{value:.0e}".replace("e-", "e−").replace("e+", "e+")
    return f"{value:g}"


def make_short_bin_labels(edges: Sequence[float]) -> List[str]:
    """Return compact display labels; exact interval rules remain in metadata."""

    labels: List[str] = []
    for lower, upper in zip(edges, edges[1:]):
        if math.isinf(upper) and upper > 0:
            labels.append(f"{_short_number(lower)}+")
        elif math.isinf(lower) and lower < 0:
            labels.append(f"<{_short_number(upper)}")
        else:
            labels.append(f"{_short_number(lower)}–{_short_number(upper)}")
    return labels


def _panel_records(
    records: Sequence[Mapping[str, object]], dataset: str, group: str, method: str
) -> Dict[int, Mapping[str, object]]:
    return {
        int(record["bin_index"]): record
        for record in records
        if record["dataset"] == dataset
        and record["group"] == group
        and record["method"] == method
    }


def canonicalize_checkpoint_values(
    datasets: Sequence[base.DiagnosticDataset],
) -> None:
    """Make each sample's fixed checkpoint values exactly identical by method.

    The base loader deliberately accepts values equal within a tight tolerance.
    Replacing those near-equal copies with the first observed pair prevents two
    methods for the same sample from landing on opposite sides of a bin edge.
    """

    for dataset in datasets:
        canonical: Dict[Tuple[str, int], Tuple[float, float]] = {}
        for row in dataset.rows:
            sample_key = (str(row["group"]), int(row["sample_index"]))
            fixed_values = canonical.setdefault(
                sample_key,
                (
                    float(row["candidate_nll"]),
                    float(row["candidate_probability"]),
                ),
            )
            row["candidate_nll"] = fixed_values[0]
            row["candidate_probability"] = fixed_values[1]


def compute_normalization_factors(
    records: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return dataset/metric/method peak-bin-median scale factors."""

    factors: Dict[str, Dict[str, Dict[str, float]]] = {}
    dataset_labels = sorted({str(record["dataset"]) for record in records})
    for dataset_label in dataset_labels:
        factors[dataset_label] = {}
        for metric, _ in base.METRICS:
            factors[dataset_label][metric] = {}
            for method in methods:
                medians = [
                    float(record[f"{metric}_median"])
                    for record in records
                    if record["dataset"] == dataset_label
                    and record["method"] == method
                    and int(record[f"{metric}_n"]) > 0
                    and record[f"{metric}_median"] is not None
                    and math.isfinite(float(record[f"{metric}_median"]))
                ]
                if not medians:
                    raise ValueError(
                        f"no non-empty bin medians for {dataset_label}/{metric}/{method}"
                    )
                factor = max(medians)
                if factor <= 0.0:
                    raise ValueError(
                        "shape normalization requires a positive peak median for "
                        f"{dataset_label}/{metric}/{method}; got {factor}"
                    )
                factors[dataset_label][metric][method] = factor
    return factors


def _scaled_value(
    record: Mapping[str, object],
    metric: str,
    statistic: str,
    normalization_factors: Optional[
        Mapping[str, Mapping[str, Mapping[str, float]]]
    ],
) -> float:
    value = float(record[f"{metric}_{statistic}"])
    if normalization_factors is None:
        return value
    factor = normalization_factors[str(record["dataset"])][metric][
        str(record["method"])
    ]
    return value / factor


def compute_shared_scales(
    records: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    normalization_factors: Optional[
        Mapping[str, Mapping[str, Mapping[str, float]]]
    ] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute one robust symlog transform and y range for each metric.

    The lower threshold uses a scale-aware floor.  This keeps exact zeros while
    preventing numerical values around 1e-22 from forcing fourteen visually
    unhelpful log decades into the full-logit panels.
    """

    selected_methods = set(methods)
    specs: Dict[str, Dict[str, float]] = {}
    for metric, _ in base.METRICS:
        values: List[float] = []
        for record in records:
            if str(record["method"]) not in selected_methods:
                continue
            if int(record[f"{metric}_n"]) == 0:
                continue
            for statistic in ("q25", "median", "q75"):
                value = record[f"{metric}_{statistic}"]
                if value is not None and math.isfinite(float(value)):
                    values.append(
                        max(
                            _scaled_value(
                                record,
                                metric,
                                statistic,
                                normalization_factors,
                            ),
                            0.0,
                        )
                    )
        if not values:
            raise ValueError(f"no finite values are available for metric {metric!r}")
        positive = [value for value in values if value > 0.0]
        maximum = max(values)
        if maximum <= 0.0:
            linthresh = 1e-12
            upper = 1.0
        else:
            upper = maximum * 1.28
            smallest_positive = min(positive) if positive else maximum
            linthresh = max(smallest_positive * 0.5, maximum * 1e-5, 1e-15)
        specs[metric] = {"linthresh": linthresh, "lower": 0.0, "upper": upper}
    return specs


def _dataset_class_count(dataset: base.DiagnosticDataset) -> int:
    if dataset.label in DATASET_CLASS_COUNTS:
        return DATASET_CLASS_COUNTS[dataset.label]
    metadata_name = str(dataset.metadata.get("dataset", "")).lower().replace("-", "")
    if metadata_name == "cifar10":
        return 10
    if metadata_name == "cifar100":
        return 100
    raise ValueError(f"cannot infer class count for dataset {dataset.label!r}")


def _value_to_bin_coordinate(
    value: float, edges: Sequence[float]
) -> Optional[float]:
    """Map a numeric threshold into a categorical-bin x coordinate."""

    if value < edges[0] or value > edges[-1]:
        return None
    number_of_bins = len(edges) - 1
    edge_array = base.np.asarray(edges, dtype=float)
    bin_index = int(base.np.searchsorted(edge_array, value, side="right") - 1)
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
    dataset: base.DiagnosticDataset, bin_by: str, edges: Sequence[float]
) -> List[Dict[str, object]]:
    class_count = _dataset_class_count(dataset)
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
            "bin_coordinate": _value_to_bin_coordinate(value, edges),
        }
        for key, label, value, linestyle, definition in specs
    ]


def plot_dataset(
    records: Sequence[Mapping[str, object]],
    dataset: base.DiagnosticDataset,
    edges: Sequence[float],
    bin_by: str,
    scales: Mapping[str, Mapping[str, float]],
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    iqr_style: str,
    flag_count: int,
    normalization_factors: Optional[
        Mapping[str, Mapping[str, Mapping[str, float]]]
    ] = None,
) -> List[Path]:
    groups = [group for group in ("difficult_clean", "noisy") if group in base.ordered_groups(dataset)]
    if len(groups) != 2:
        raise ValueError(
            f"{dataset.label} must contain difficult_clean and noisy groups; got {groups}"
        )

    number_of_bins = len(edges) - 1
    x = base.np.arange(number_of_bins)
    tick_labels = make_short_bin_labels(edges)
    threshold_specs = reference_thresholds(dataset, bin_by, edges)
    normalized = normalization_factors is not None
    x_axis_label = (
        r"Candidate label bins with $-\log p_y$"
        if bin_by == "nll"
        else r"Checkpoint candidate-label probability $p_y$ bin"
    )
    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }

    with base.plt.rc_context(rc):
        figure, axes = base.plt.subplots(
            2,
            2,
            figsize=(11.0, 7.15),
            squeeze=False,
            sharex="col",
            sharey="col",
        )
        for row_index, group in enumerate(groups):
            for column_index, (metric, metric_label) in enumerate(base.METRICS):
                axis = axes[row_index, column_index]
                count_vectors = []
                for method in SPLIT_METHODS:
                    panel = _panel_records(records, dataset.label, group, method)
                    counts = base.np.zeros(number_of_bins, dtype=int)
                    for bin_index in range(number_of_bins):
                        record = panel.get(bin_index)
                        if record is not None:
                            counts[bin_index] = int(record[f"{metric}_n"])
                    count_vectors.append(counts)
                minimum_method_counts = base.np.min(
                    base.np.vstack(count_vectors), axis=0
                )
                low_count_bins = (
                    (minimum_method_counts > 0)
                    & (minimum_method_counts < flag_count)
                )

                for method in SPLIT_METHODS:
                    panel = _panel_records(records, dataset.label, group, method)
                    medians = base.np.full(number_of_bins, base.np.nan)
                    q25 = base.np.full(number_of_bins, base.np.nan)
                    q75 = base.np.full(number_of_bins, base.np.nan)
                    for bin_index in range(number_of_bins):
                        record = panel.get(bin_index)
                        if record is None or int(record[f"{metric}_n"]) == 0:
                            continue
                        medians[bin_index] = _scaled_value(
                            record, metric, "median", normalization_factors
                        )
                        q25[bin_index] = _scaled_value(
                            record, metric, "q25", normalization_factors
                        )
                        q75[bin_index] = _scaled_value(
                            record, metric, "q75", normalization_factors
                        )

                    finite_band = base.np.isfinite(q25) & base.np.isfinite(q75)
                    finite_line = base.np.isfinite(medians)
                    if not base.np.any(finite_line):
                        continue
                    color = base.METHOD_COLORS[method]
                    if iqr_style == "band":
                        axis.fill_between(
                            x,
                            q25,
                            q75,
                            where=finite_band,
                            color=color,
                            alpha=0.09,
                            linewidth=0.0,
                        )
                    elif iqr_style == "whisker":
                        axis.errorbar(
                            x[finite_band],
                            medians[finite_band],
                            yerr=base.np.vstack(
                                (
                                    medians[finite_band] - q25[finite_band],
                                    q75[finite_band] - medians[finite_band],
                                )
                            ),
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.0,
                            capsize=2.2,
                            capthick=0.9,
                            alpha=0.38,
                            zorder=2,
                        )
                    axis.plot(
                        x,
                        medians,
                        color=color,
                        marker=base.METHOD_MARKERS[method],
                        markersize=4.3,
                        linewidth=1.9,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=0.55,
                        label=SPLIT_METHOD_LABELS[method],
                        zorder=3,
                    )
                    low_count = finite_line & low_count_bins
                    if flag_count > 0 and base.np.any(low_count):
                        axis.plot(
                            x[low_count],
                            medians[low_count],
                            linestyle="none",
                            marker=base.METHOD_MARKERS[method],
                            markersize=4.6,
                            markerfacecolor="white",
                            markeredgecolor=color,
                            markeredgewidth=1.15,
                            zorder=4,
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
                        fontsize=7,
                        color="#525252",
                    )

                scale = scales[metric]
                axis.set_yscale(
                    "symlog", linthresh=scale["linthresh"], linscale=0.75
                )
                axis.set_ylim(scale["lower"], scale["upper"])
                axis.set_xlim(-0.35, number_of_bins - 0.65)
                axis.axhline(0.0, color="#737373", linewidth=0.55, alpha=0.6)
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
                if row_index == 1:
                    axis.set_xticklabels(tick_labels, rotation=30, ha="right")
                else:
                    axis.tick_params(axis="x", labelbottom=False)
                if row_index == 0:
                    axis.set_title(metric_label, pad=8)
                axis.text(
                    0.98,
                    0.96,
                    base.GROUP_LABELS.get(group, group),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9.5,
                    #fontweight="semibold",
                )

        legend_handles = [
            base.Line2D(
                [0],
                [0],
                color=base.METHOD_COLORS[method],
                marker=base.METHOD_MARKERS[method],
                markersize=5,
                linewidth=1.9,
                markerfacecolor=base.METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.55,
                label=SPLIT_METHOD_LABELS[method],
            )
            for method in SPLIT_METHODS
        ]
        mode_title = "Shape-normalized" if normalized else "Raw absolute"
        figure.suptitle(
            f"{mode_title} gradient response under {dataset.label}",
            fontsize=12,
            # fontweight="semibold",
            y=0.998,
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.982),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.8,
        )
        if flag_count > 0:
            low_count_handle = base.Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor="#555555",
                markeredgewidth=1.15,
                label=rf"Hollow marker: bin sample count $0 < n < {flag_count}$",
            )
            figure.legend(
                handles=[low_count_handle],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.944),
                frameon=False,
                fontsize=7.8,
                handletextpad=0.55,
            )
        figure.supxlabel(x_axis_label, fontsize=10, y=0.082)
        y_axis_label = (
            "Normalized gradient response (log)"
            if normalized
            else "Absolute sample gradient magnitude (symlog)"
        )
        figure.supylabel(y_axis_label, fontsize=9.5, x=0.012)
        if iqr_style == "whisker":
            spread_note = "Median with q25–q75 whiskers"
        elif iqr_style == "band":
            spread_note = "Median with interquartile band"
        else:
            spread_note = "Median only"
        class_count = _dataset_class_count(dataset)
        if bin_by == "nll":
            c_value = math.log(class_count)
            reference_note = (
                rf"Vertical references: $c_C=\ln({class_count})={c_value:.6g}$ "
                rf"(dashed); $2c_C={2.0 * c_value:.6g}$ (dotted)."
            )
        else:
            reference_note = (
                rf"Vertical references: $1/C={1.0 / class_count:.6g}$ (dashed); "
                rf"$1/C^2={1.0 / (class_count * class_count):.6g}$ (dotted)."
            )
        normalization_note = (
            "Each method is divided by its dataset/metric peak bin median across "
            "both groups."
            if normalized
            else "Absolute magnitudes are shown without method-wise rescaling."
        )
        # figure.text(
        #     0.5,
        #     0.010,
        #     "Fixed-checkpoint instantaneous logit response; SADT active is the "
        #     "active branch, while Full SADT is the complete loss.\n"
        #     f"{spread_note}; hollow markers indicate "
        #     rf"$0 < \min_m n_m < {flag_count}$. "
        #     f"{normalization_note}\n{reference_note}",
        #     ha="center",
        #     va="bottom",
        #     fontsize=7.8,
        #     color="#4A4A4A",
        # )
        figure.subplots_adjust(
            left=0.095,
            right=0.985,
            bottom=0.185,
            top=0.855 if flag_count > 0 else 0.895,
            hspace=0.23,
            wspace=0.16,
        )

        written: List[Path] = []
        for file_format in formats:
            output_path = output_stem.with_suffix(f".{file_format}")
            save_options = {"bbox_inches": "tight"}
            if file_format in {"png", "tif", "tiff"}:
                save_options["dpi"] = dpi
            if file_format in {"tif", "tiff"}:
                save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
            figure.savefig(output_path, **save_options)
            written.append(output_path)
        base.plt.close(figure)
    return written


def write_metadata(
    output_path: Path,
    datasets: Sequence[base.DiagnosticDataset],
    edges: Sequence[float],
    edge_source: str,
    bin_by: str,
    raw_scales_by_dataset: Mapping[str, Mapping[str, Mapping[str, float]]],
    normalized_scales_by_dataset: Mapping[str, Mapping[str, Mapping[str, float]]],
    normalization_factors: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ],
    iqr_style: str,
    flag_count: int,
    outputs: Sequence[Path],
) -> None:
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "name": script_path.name,
            "version": SCRIPT_VERSION,
            "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "execution": "CPU-only post-processing; no model forward/backward pass",
        "binning": {
            "selector": bin_by,
            "source_column": base.BIN_COLUMNS[bin_by],
            "edge_source": edge_source,
            "edges": [base.json_edge(value) for value in edges],
            "interval_rule": (
                "left-closed/right-open [lower, upper), except a finite final "
                "upper edge is included"
            ),
            "shared_across_all_datasets": True,
            "shared_across_all_methods": True,
            "checkpoint_value_canonicalization": (
                "within each dataset and (group, sample_index), all methods use "
                "the first observed candidate_nll and candidate_probability pair"
            ),
        },
        "statistics": {
            "center": "median",
            "iqr": "q25-q75",
            "iqr_style": iqr_style,
            "methods": list(SPLIT_METHODS),
            "method_display_labels": SPLIT_METHOD_LABELS,
        },
        "shape_normalization": {
            "definition": (
                "for each dataset x metric x method, divide median, q25, and q75 "
                "by the maximum non-empty bin median across difficult_clean and "
                "noisy groups"
            ),
            "factors": normalization_factors,
        },
        "figures": {
            "layout_per_dataset": "2 rows (difficult_clean/noisy) x 2 metrics",
            "y_scale": (
                "symlog; each dataset figure has one metric-specific transform "
                "and limit set shared across its two rows"
            ),
            "raw_shared_scales_by_dataset": raw_scales_by_dataset,
            "normalized_shared_scales_by_dataset": normalized_scales_by_dataset,
            "short_bin_labels": make_short_bin_labels(edges),
            "marker_style": (
                "filled method marker by default; hollow marker when the minimum "
                "count across methods in the same dataset/group/metric/bin is "
                "positive and below the configured threshold; empty bins are omitted"
            ),
            "small_bin_flag": {
                "enabled": flag_count > 0,
                "threshold": flag_count,
                "rule": "0 < min_m n_m < threshold",
                "legend_label": (
                    f"Hollow marker: 0 < min_m n_m < {flag_count}"
                    if flag_count > 0
                    else None
                ),
            },
            "reference_lines": {
                dataset.label: {
                    "class_count": _dataset_class_count(dataset),
                    "coordinate_rule": (
                        "linear interpolation between the left and right categorical "
                        "boundaries of the containing numeric bin; an unbounded bin "
                        "uses its center"
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
                        for threshold in reference_thresholds(dataset, bin_by, edges)
                    ],
                }
                for dataset in datasets
            },
        },
        "inputs": [
            {
                "label": dataset.label,
                "directory": str(dataset.source_dir),
                "sample_gradient_rows": len(dataset.rows),
                "metadata_dataset": dataset.metadata.get("dataset"),
                "noise_type": dataset.metadata.get("noise_type"),
                "noise_rate": dataset.metadata.get("noise_rate"),
            }
            for dataset in datasets
        ],
        "outputs": [path.name for path in outputs],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.flag_count < 0:
        raise SystemExit("--flag-count must be nonnegative")
    try:
        datasets = [
            load_split_diagnostic("CIFAR-10N with worst labels", args.c10n_dir),
            load_split_diagnostic("CIFAR-100N with noisy labels", args.c100n_dir),
        ]
        canonicalize_checkpoint_values(datasets)
        edges, edge_source = base.parse_edges(args.edges, args.bin_by)
        formats = base.parse_formats(args.formats)
        base.assign_bins(datasets, args.bin_by, edges)
        summary = base.build_summary(datasets, args.bin_by, edges)
        normalization_factors = compute_normalization_factors(summary, SPLIT_METHODS)
        raw_scales_by_dataset = {
            dataset.label: compute_shared_scales(
                [record for record in summary if record["dataset"] == dataset.label],
                SPLIT_METHODS,
            )
            for dataset in datasets
        }
        normalized_scales_by_dataset = {
            dataset.label: compute_shared_scales(
                [record for record in summary if record["dataset"] == dataset.label],
                SPLIT_METHODS,
                normalization_factors,
            )
            for dataset in datasets
        }
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"input validation failed: {exc}") from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "gradient_bin_summary_split.csv"
    metadata_path = output_dir / "split_binning_metadata.json"
    base.write_summary_csv(summary, summary_path)

    figure_paths: List[Path] = []
    for dataset in datasets:
        figure_paths.extend(
            plot_dataset(
                summary,
                dataset,
                edges,
                args.bin_by,
                raw_scales_by_dataset[dataset.label],
                output_dir / OUTPUT_STEMS[dataset.label],
                formats,
                args.dpi,
                args.iqr_style,
                args.flag_count,
            )
        )
        figure_paths.extend(
            plot_dataset(
                summary,
                dataset,
                edges,
                args.bin_by,
                normalized_scales_by_dataset[dataset.label],
                output_dir / f"{OUTPUT_STEMS[dataset.label]}_normalized",
                formats,
                args.dpi,
                args.iqr_style,
                args.flag_count,
                normalization_factors,
            )
        )

    all_outputs = [summary_path, metadata_path] + figure_paths
    write_metadata(
        metadata_path,
        datasets,
        edges,
        edge_source,
        args.bin_by,
        raw_scales_by_dataset,
        normalized_scales_by_dataset,
        normalization_factors,
        args.iqr_style,
        args.flag_count,
        all_outputs,
    )

    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Recorded fixed bins, normalization factors, and scales in {metadata_path}")
    for path in figure_paths:
        print(f"Wrote figure {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
