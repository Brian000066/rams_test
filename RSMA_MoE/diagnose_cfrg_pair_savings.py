"""Diagnose CFRG pair savings before the interval-grouping DP runs.

The script executes SCOPE and CFRG IoT selection, enumerates every selected-IoT
pair that has a feasible common receiving server, records bandwidth/cost savings,
and stops at the DP entry point.  It does not change the scheduling algorithms.

Outputs:
  * pair_records.csv: one row per feasible pair
  * per_run_summary.csv: pair counts and positive-saving rates per run
  * summary.json: pooled statistics and MST-contiguity coverage
  * figure/delta_bandwidth_histogram.png
  * figure/delta_cost_histogram.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import replace
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ExperimentConfig
from experiment_runner import RunSeeds, generate_task_graphs, run_enabled_methods
from our_alg.phase_2 import Phase2GroupingBackhaulMixin


class _DiagnosticStop(RuntimeError):
    """Stop the pipeline immediately before the CFRG interval DP."""


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return (
            ordered[lower] * (upper - position)
            + ordered[upper] * (position - lower)
        )

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_histogram(
    values: Sequence[float],
    *,
    path: Path,
    title: str,
    xlabel: str,
    positive_probability: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(float(value) for value in values)
    limits = _quantiles(ordered)
    lower = limits["p10"] if len(ordered) < 100 else ordered[max(0, len(ordered) // 100)]
    upper_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.99) - 1)
    upper = ordered[upper_index]
    central = [value for value in ordered if lower <= value <= upper]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    full_axis, zoom_axis = axes
    full_axis.hist(
        ordered,
        bins=60,
        color="#4C78A8",
        alpha=0.85,
        edgecolor="white",
    )
    full_axis.axvline(0.0, color="#E45756", linestyle="--", linewidth=2, label="zero")
    full_axis.set_yscale("log")
    full_axis.set_title("Full range (log count)")

    zoom_axis.hist(
        central,
        bins=60,
        color="#72B7B2",
        alpha=0.9,
        edgecolor="white",
    )
    zoom_axis.axvline(0.0, color="#E45756", linestyle="--", linewidth=2, label="zero")
    zoom_axis.set_title(
        f"Central range ({len(central)}/{len(ordered)} pairs shown)"
    )

    figure.suptitle(f"{title}\nPr(value > 0) = {positive_probability:.2%}")
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of feasible pairs")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_diagnostic(
    *,
    runs: int,
    dag_seed: int,
    network_seed: int,
    output_dir: Path,
    tolerance: float = 1e-9,
) -> Path:
    if runs <= 0:
        raise ValueError("runs must be positive")

    config = replace(
        ExperimentConfig(),
        auto_random_seed=False,
        run_hybrid_topk_gssgd_backhaul=False,
        run_hybrid_topk_location_aware_backhaul=False,
        run_wdmoe_gssgd=False,
        run_wdmoe_location_aware=False,
        run_our_similarity_aware_topk_backhaul=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_interval_groups = Phase2GroupingBackhaulMixin._cfrg_interval_groups
    pair_records: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []
    context: Dict[str, int] = {}

    def diagnostic_hook(self, order, ownership):
        position = {device_id: index for index, device_id in enumerate(order)}
        all_pair_count = len(order) * (len(order) - 1) // 2
        current_records: List[Dict[str, Any]] = []

        for left, right in combinations(order, 2):
            pair_info = self._cfrg_interval_info((left, right), ownership)
            if not pair_info.get("feasible"):
                continue
            left_info = self._cfrg_interval_info((left,), ownership)
            right_info = self._cfrg_interval_info((right,), ownership)
            if not left_info.get("feasible") or not right_info.get("feasible"):
                continue

            left_bandwidth = float(left_info["bandwidth"])
            right_bandwidth = float(right_info["bandwidth"])
            pair_bandwidth = float(pair_info["bandwidth"])
            delta_bandwidth = left_bandwidth + right_bandwidth - pair_bandwidth

            singleton_forwarding = (
                float(left_info["forwarding_cost"])
                + float(right_info["forwarding_cost"])
            )
            pair_forwarding = float(pair_info["forwarding_cost"])
            singleton_cost = (
                self.config.c_bw * (left_bandwidth + right_bandwidth)
                + singleton_forwarding
            )
            pair_cost = self.config.c_bw * pair_bandwidth + pair_forwarding
            delta_cost = singleton_cost - pair_cost
            dissimilarity, minimizing_server = self._cfrg_pair_dissimilarity(
                left,
                right,
            )

            record = {
                "run": context["run"],
                "dag_seed": context["dag_seed"],
                "network_seed": context["network_seed"],
                "left_iot": left,
                "right_iot": right,
                "left_order_index": position[left],
                "right_order_index": position[right],
                "mst_contiguous": abs(position[left] - position[right]) == 1,
                "pair_origin_server": pair_info["origin_server"],
                "left_origin_server": left_info["origin_server"],
                "right_origin_server": right_info["origin_server"],
                "pair_relation_minimizing_server": minimizing_server,
                "pair_dissimilarity": dissimilarity,
                "left_bandwidth_hz": left_bandwidth,
                "right_bandwidth_hz": right_bandwidth,
                "pair_bandwidth_hz": pair_bandwidth,
                "delta_bandwidth_hz": delta_bandwidth,
                "bandwidth_saving_positive": delta_bandwidth > tolerance,
                "singleton_forwarding_cost": singleton_forwarding,
                "pair_forwarding_cost": pair_forwarding,
                "singleton_bw_plus_fwd_cost": singleton_cost,
                "pair_bw_plus_fwd_cost": pair_cost,
                "delta_bw_plus_fwd_cost": delta_cost,
                "total_cost_saving_positive": delta_cost > tolerance,
                "pair_time_budget_seconds": pair_info["time_budget"],
                "pair_worst_channel_gain": pair_info["worst_channel_gain"],
                "pair_required_feature_count": len(pair_info["required_features"]),
                "pair_destination_server_count": len(pair_info["destinations"]),
            }
            current_records.append(record)
            pair_records.append(record)

        feasible_count = len(current_records)
        bandwidth_positive = sum(
            bool(row["bandwidth_saving_positive"])
            for row in current_records
        )
        cost_positive = sum(
            bool(row["total_cost_saving_positive"])
            for row in current_records
        )
        bandwidth_positive_adjacent = sum(
            bool(row["bandwidth_saving_positive"] and row["mst_contiguous"])
            for row in current_records
        )
        cost_positive_adjacent = sum(
            bool(row["total_cost_saving_positive"] and row["mst_contiguous"])
            for row in current_records
        )
        run_summaries.append(
            {
                "run": context["run"],
                "dag_seed": context["dag_seed"],
                "network_seed": context["network_seed"],
                "selected_iots": len(order),
                "all_selected_iot_pairs": all_pair_count,
                "feasible_pairs": feasible_count,
                "bandwidth_positive_pairs": bandwidth_positive,
                "bandwidth_positive_probability": (
                    bandwidth_positive / feasible_count if feasible_count else 0.0
                ),
                "cost_positive_pairs": cost_positive,
                "cost_positive_probability": (
                    cost_positive / feasible_count if feasible_count else 0.0
                ),
                "bandwidth_positive_mst_contiguous": bandwidth_positive_adjacent,
                "cost_positive_mst_contiguous": cost_positive_adjacent,
            }
        )
        raise _DiagnosticStop

    Phase2GroupingBackhaulMixin._cfrg_interval_groups = diagnostic_hook
    try:
        for run_index in range(runs):
            seeds = RunSeeds(
                dag_seed=dag_seed + run_index,
                network_seed=network_seed + run_index,
            )
            run_config = replace(
                config,
                random_seed=seeds.dag_seed,
                network_random_seed=seeds.network_seed,
                results_dir=output_dir / "_working" / f"run_{run_index + 1}",
            )
            run_config.results_dir.mkdir(parents=True, exist_ok=True)
            graphs = generate_task_graphs(
                run_config,
                seeds,
                verbose=False,
                export_artifacts=False,
            )
            context.clear()
            context.update(
                run=run_index + 1,
                dag_seed=seeds.dag_seed,
                network_seed=seeds.network_seed,
            )
            try:
                run_enabled_methods(
                    run_config,
                    graphs,
                    seeds,
                    print_details=False,
                    show_plot=False,
                )
            except _DiagnosticStop:
                pass
            else:
                raise RuntimeError("diagnostic hook did not stop before the DP")

            summary = run_summaries[-1]
            print(
                f"Run {run_index + 1}/{runs}: "
                f"selected={summary['selected_iots']}, "
                f"feasible_pairs={summary['feasible_pairs']}, "
                f"Pr(delta_B>0)={summary['bandwidth_positive_probability']:.2%}, "
                f"Pr(delta_C>0)={summary['cost_positive_probability']:.2%}"
            )
    finally:
        Phase2GroupingBackhaulMixin._cfrg_interval_groups = original_interval_groups

    if not pair_records:
        raise RuntimeError("no feasible selected-IoT pairs were found")

    delta_bandwidths = [
        float(row["delta_bandwidth_hz"])
        for row in pair_records
    ]
    delta_costs = [
        float(row["delta_bw_plus_fwd_cost"])
        for row in pair_records
    ]
    positive_bandwidth_records = [
        row for row in pair_records if row["bandwidth_saving_positive"]
    ]
    positive_cost_records = [
        row for row in pair_records if row["total_cost_saving_positive"]
    ]
    bandwidth_adjacent = sum(
        bool(row["mst_contiguous"])
        for row in positive_bandwidth_records
    )
    cost_adjacent = sum(
        bool(row["mst_contiguous"])
        for row in positive_cost_records
    )

    summary = {
        "settings": {
            "runs": runs,
            "base_dag_seed": dag_seed,
            "base_network_seed": network_seed,
            "positive_tolerance": tolerance,
            "dp_executed": False,
            "pair_definition": (
                "selected IoT pair with at least one common reachable server; "
                "the current CFRG delivery rule selects the pair origin"
            ),
            "delta_bandwidth": "B(singleton_i)+B(singleton_j)-B(pair)",
            "delta_cost": (
                "C_bw+forwarding(singleton_i)+C_bw+forwarding(singleton_j)"
                "-C_bw+forwarding(pair)"
            ),
        },
        "selected_iots": {
            "mean_per_run": statistics.fmean(
                int(row["selected_iots"])
                for row in run_summaries
            ),
            "min_per_run": min(int(row["selected_iots"]) for row in run_summaries),
            "max_per_run": max(int(row["selected_iots"]) for row in run_summaries),
        },
        "pairs": {
            "all_selected_iot_pairs": sum(
                int(row["all_selected_iot_pairs"])
                for row in run_summaries
            ),
            "feasible_pairs": len(pair_records),
        },
        "delta_bandwidth": {
            "positive_pairs": len(positive_bandwidth_records),
            "nonpositive_pairs": len(pair_records) - len(positive_bandwidth_records),
            "positive_probability": len(positive_bandwidth_records) / len(pair_records),
            "quantiles_hz": _quantiles(delta_bandwidths),
        },
        "delta_cost": {
            "positive_pairs": len(positive_cost_records),
            "nonpositive_pairs": len(pair_records) - len(positive_cost_records),
            "positive_probability": len(positive_cost_records) / len(pair_records),
            "quantiles": _quantiles(delta_costs),
        },
        "mst_contiguous_restriction": {
            "positive_bandwidth_pairs_contiguous": bandwidth_adjacent,
            "positive_bandwidth_pairs_noncontiguous": (
                len(positive_bandwidth_records) - bandwidth_adjacent
            ),
            "positive_bandwidth_pair_capture_probability": (
                bandwidth_adjacent / len(positive_bandwidth_records)
                if positive_bandwidth_records else 0.0
            ),
            "positive_cost_pairs_contiguous": cost_adjacent,
            "positive_cost_pairs_noncontiguous": (
                len(positive_cost_records) - cost_adjacent
            ),
            "positive_cost_pair_capture_probability": (
                cost_adjacent / len(positive_cost_records)
                if positive_cost_records else 0.0
            ),
        },
    }

    _write_csv(output_dir / "pair_records.csv", pair_records)
    _write_csv(output_dir / "per_run_summary.csv", run_summaries)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    _plot_histogram(
        [value / 1e6 for value in delta_bandwidths],
        path=output_dir / "figure" / "delta_bandwidth_histogram.png",
        title="CFRG Pair Bandwidth Saving Before DP",
        xlabel="Delta B = B_i + B_j - B_{i,j} (MHz)",
        positive_probability=summary["delta_bandwidth"]["positive_probability"],
    )
    _plot_histogram(
        delta_costs,
        path=output_dir / "figure" / "delta_cost_histogram.png",
        title="CFRG Pair Bandwidth + Forwarding Cost Saving Before DP",
        xlabel="Delta C = C_i + C_j - C_{i,j}",
        positive_probability=summary["delta_cost"]["positive_probability"],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output directory: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--dag-seed", type=int, default=42)
    parser.add_argument("--network-seed", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Optional[Path] = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / "cfrg_pair_diagnostic" / timestamp
    run_diagnostic(
        runs=args.runs,
        dag_seed=args.dag_seed,
        network_seed=args.network_seed,
        output_dir=output_dir,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
