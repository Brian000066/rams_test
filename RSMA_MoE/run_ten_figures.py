"""Sensitivity figures 1-13 for the JRGXP paper (line charts).

Just run::

    python3 run_ten_figures.py

This collects 300 accepted runs per x value (an accepted run is one in which
all five methods have zero violations; seeds are deterministic:
dag_seed = 42 + i, network_seed = 1 + i), several figures in parallel, and
writes the line charts to results/ten_figures/<time>/figures/.

Plotting (default): runs in which any method exceeds 10 x its own median at
that x value are dropped for all methods (listed in removed_outliers.csv), and
each point is the mean of the first 100 remaining runs.

Fig. 3 (MEC servers 10-30) is nested and paired: the first 10 servers, the
IoTs and the tasks are identical for every server count (larger counts only
add servers), every count runs the same seeds, and only seeds accepted at all
counts are averaged.  Fig. 4 uses average deadlines 2, 4, 6, 8, and 10 s,
drawn from U[avg-0.5, avg+0.5] s, with a 0.4 Mbit/s minimum IoT rate and
unchanged cost weights. It is
paired as well (same seeds at every deadline).
Fig. 7 (experts stored per server) is also nested and paired: every average
keeps the 10-14 deployment and only adds replicas, and every subtask has a
fixed 12 gating candidates, so tasks are identical across the x values.
Fig. 12 plots the RSMA group counts recorded by Fig. 2, so it needs no
additional simulation runs.
Fig. 13 plots the bandwidth costs recorded by Fig. 4 against the same
deadlines, so it also needs no additional simulation runs.

Every attempt is saved immediately; running the same command again continues
an unfinished run.  Options: --runs N (simulation), --plot-runs N (0 = all),
--keep-outliers, --outlier-factor F, --figures 1,2,5, --workers N, --new,
--resume <run_dir>, --plot-only <run_dir> (redraw only), --quick.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import ExperimentConfig  # noqa: E402
from experiment_runner import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    generate_task_graphs,
    resolve_run_seeds,
    run_enabled_methods,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "results" / "ten_figures"

METHOD_ORDER = (
    "hybrid_topk_gssgd_backhaul",
    "hybrid_topk_location_aware_backhaul",
    "wdmoe_gssgd",
    "wdmoe_location_aware",
    "our_similarity_aware_topk_backhaul",
)
METHOD_STYLE = {
    # (color, marker, linestyle) in the style of the SIoT paper figures
    "hybrid_topk_gssgd_backhaul": ("#32CD32", "o", "-"),
    "hybrid_topk_location_aware_backhaul": ("#FF8C00", "s", "--"),
    "wdmoe_gssgd": ("#DC143C", "D", "-."),
    "wdmoe_location_aware": ("#8A2BE2", "v", (0, (5, 1, 1, 1, 1, 1))),
    "our_similarity_aware_topk_backhaul": ("#4169E1", "^", ":"),
}
PLOT_LABELS = {
    "hybrid_topk_gssgd_backhaul": "Top-k+CSSGD",
    "hybrid_topk_location_aware_backhaul": "Top-k+Location-aware",
    "wdmoe_gssgd": "WDMoE+CSSGD",
    "wdmoe_location_aware": "WDMoE+Location-aware",
    "our_similarity_aware_topk_backhaul": "RESG-GoT",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
})


def _avg_range(avg: int) -> tuple[int, int]:
    """Integer range [avg-1, avg+1], whose mean is avg."""
    return (int(avg) - 1, int(avg) + 1)


def _half_range(avg: float) -> tuple[float, float]:
    """Deadline range [avg-0.5, avg+0.5] s (continuous uniform), whose mean is avg."""
    return (float(avg) - 0.5, float(avg) + 0.5)


def _set_deadline_sensitivity(config: ExperimentConfig, average: float) -> ExperimentConfig:
    """Set the deadline range and minimum IoT rate for Figure 4."""
    return replace(
        config,
        task_deadline_seconds_range=_half_range(average),
        min_rate=0.4e6,
    )


NESTED_BASE_SERVERS = 10


def _set_servers_nested(config: ExperimentConfig, servers: int) -> ExperimentConfig:
    """Nested design: the first NESTED_BASE_SERVERS servers, IoTs and tasks are
    identical for every server count; larger counts only add servers."""
    return replace(config, num_edge_servers=int(servers), nested_server_base=NESTED_BASE_SERVERS)


def _avg_range_wide(avg: int) -> tuple[int, int]:
    """Integer range [avg-2, avg+2] (used for experts stored per server)."""
    return (int(avg) - 2, int(avg) + 2)


FIG7_GATING_CANDIDATES = 12
FIG7_BASE_AVG = 12


def _set_experts_per_server(config: ExperimentConfig, avg: int) -> ExperimentConfig:
    """Stored experts per server ~ U[avg-2, avg+2].

    GPU memory is scaled with the average number of stored experts
    (default 300-380 MB for the default 10-15 experts, avg 12.5) so that the
    fraction of stored experts that can be active at once stays the same.
    Without scaling, Top-K cannot place 5 experts at avg 16-18 and almost no
    attempt has zero violations for all five methods.
    """
    scale = float(avg) / 12.5
    low, high = config.server_gpu_memory_range
    return replace(
        config,
        experts_per_server=_avg_range_wide(avg),
        server_gpu_memory_range=(low * scale, high * scale),
        # Fixed number of gating candidates, so gating does not get flatter
        # as servers store more experts.
        gating_num_candidates=FIG7_GATING_CANDIDATES,
        # Nested storage: every average keeps the 10-14 deployment and only
        # adds replicas, so the same seed gives identical tasks.
        nested_storage_base=_avg_range_wide(FIG7_BASE_AVG),
    )


def _set_dag_nodes(config: ExperimentConfig, nodes: int) -> ExperimentConfig:
    """Subtasks per task.  The maximum branch length grows with the node count
    (5 at 10 nodes, i.e. ceil(nodes / 2)) so that 2-4 branches can hold up to
    30 nodes; 10 nodes keeps the default (1, 5)."""
    spec = config.common_dag_spec
    low, high = spec.branch_length_range or (1, 5)
    high = max(int(high), math.ceil(int(nodes) / 2))
    return replace(config, common_dag_spec=replace(spec, num_nodes=int(nodes), branch_length_range=(low, high)))


def _set_required_features_per_task(config: ExperimentConfig, average: float) -> ExperimentConfig:
    """Set the expected sum of required feature counts across one task's subtasks.

    Each subtask independently draws an integer count from the configured
    inclusive range.  Figure 9 uses targets whose per-subtask averages are
    integers or half-integers, so a one- or two-value uniform range gives the
    requested task-level average without changing the DAG generator.
    """
    per_subtask = float(average) / config.common_dag_spec.num_nodes
    low, high = math.floor(per_subtask), math.ceil(per_subtask)
    if low < 0 or not math.isclose((low + high) / 2, per_subtask):
        raise ValueError("Task feature average must map to an integer or half-integer per subtask.")
    return replace(config, iot_features_per_node_range=(low, high))


@dataclass(frozen=True)
class Figure:
    number: int
    key: str
    values: tuple[float, ...]
    apply: Callable[[ExperimentConfig, float], ExperimentConfig]
    metric: str
    x_label: str
    y_label: str
    x_tick_labels: tuple[str, ...] | None = None
    # Paired: every x value runs the same attempts (seeds); only seeds that
    # are accepted at all x values are averaged.
    paired: bool = False
    # Logarithmic y axis (e.g. bandwidth cost spanning several orders of magnitude).
    log_y: bool = False
    # Reuse another figure's observations when the x sweep is identical.
    source_figure_number: int | None = None


FIGURES: tuple[Figure, ...] = (
    Figure(1, "num_tasks", (10, 15, 20, 25, 30),
           lambda c, v: replace(c, num_dags=int(v)),
           "total_cost", "Number of tasks", "Total cost"),
    Figure(2, "num_iots", (120, 140, 160, 180, 200),
           lambda c, v: replace(c, num_iot_devices=int(v)),
           "total_cost", "Number of IoTs", "Total cost"),
    Figure(3, "num_servers", (10, 15, 20, 25, 30),
           _set_servers_nested,
           "total_cost", "Number of MEC servers", "Total cost", paired=True),
    Figure(4, "deadline_2to10_rate04", (2, 4, 6, 8, 10),
           _set_deadline_sensitivity,
           "communication_cost", "Average deadline (s)",
           "Communication cost (bandwidth + forwarding)", paired=True),
    Figure(5, "max_group_size", (2, 3, 4, 5, 6),
           lambda c, v: replace(c, max_group_size=int(v)),
           "bandwidth_cost", r"Maximum group size $N_{th}$", "Bandwidth cost"),
    Figure(6, "subtasks_per_task", (10, 15, 20, 25, 30),
           _set_dag_nodes,
           "total_cost", "Number of subtasks per task", "Total cost"),
    Figure(7, "experts_per_server", (12, 16, 20, 24, 28),
           _set_experts_per_server,
           "total_cost", "Average number of experts stored per server", "Total cost", paired=True),
    Figure(8, "features_per_iot", (10, 15, 20, 25, 30),
           lambda c, v: replace(c, iot_features_per_device_range=_avg_range(v)),
           "communication_cost", "Average number of features per IoT",
           "Communication cost (bandwidth + forwarding)"),
    Figure(9, "required_features_per_task", (40, 50, 60, 70, 80),
           _set_required_features_per_task,
           "total_cost", "Average number of required features per task", "Total cost"),
    Figure(10, "shared_kinds", (0, 2, 4, 6, 8),
           lambda c, v: replace(c, shared_subtask_kind_count=int(v)),
           "total_cost", "Number of shared subtasks (kinds)", "Total cost"),
)
FIGURES = FIGURES + (
    Figure(11, "multi_iot_groups", (2, 3, 4, 5, 6),
           lambda c, v: replace(c, max_group_size=int(v)),
           "multi_groups", r"Maximum group size $N_{th}$",
           "Number of groups $-$ number of single-IoT groups"),
    Figure(12, "num_iots_rsma_groups", (120, 140, 160, 180, 200),
           lambda c, v: replace(c, num_iot_devices=int(v)),
           "num_groups", "Number of IoTs", "Number of RSMA groups",
           source_figure_number=2),
    Figure(13, "deadline_bandwidth", (2, 4, 6, 8, 10),
           _set_deadline_sensitivity,
           "bandwidth_cost", "Average deadline (s)", "Bandwidth cost",
           paired=True, source_figure_number=4),
)
FIGURE_BY_NUMBER = {figure.number: figure for figure in FIGURES}

FIELDS = (
    "figure", "key", "x", "attempt", "accepted_index", "dag_seed", "network_seed",
    "method_key", "total_cost", "activation_cost", "bandwidth_cost",
    "forwarding_cost", "inference_cost", "communication_cost",
    "num_groups", "single_groups", "multi_groups",
)
ATTEMPT_FIELDS = ("figure", "key", "x", "attempt", "accepted", "seconds", "violations")


def _record(result: Any) -> dict[str, float]:
    bandwidth = float(getattr(result, "bandwidth_cost", 0.0))
    forwarding = float(getattr(result, "forwarding_cost", 0.0))
    group_sizes = [
        len(members)
        for members_list in (getattr(result, "rsma_groups", {}) or {}).values()
        for members in members_list
    ]
    num_groups = len(group_sizes)
    single_groups = sum(1 for size in group_sizes if size == 1)
    return {
        "total_cost": float(getattr(result, "total_cost", 0.0)),
        "activation_cost": float(getattr(result, "activation_cost", 0.0)),
        "bandwidth_cost": bandwidth,
        "forwarding_cost": forwarding,
        "inference_cost": float(getattr(result, "inference_cost", 0.0)),
        "communication_cost": bandwidth + forwarding,
        "num_groups": num_groups,
        "single_groups": single_groups,
        "multi_groups": num_groups - single_groups,
    }


def _figure_dir(run_dir: Path, figure: Figure) -> Path:
    return run_dir / f"fig{figure.number:02d}_{figure.key}"


def _load_progress(fig_dir: Path) -> tuple[dict, dict]:
    """Return accepted counts and last attempt number per x value."""
    accepted: dict[float, int] = {}
    last_attempt: dict[float, int] = {}
    attempts_path = fig_dir / "attempts.csv"
    if attempts_path.exists():
        with attempts_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                x = float(row["x"])
                last_attempt[x] = max(last_attempt.get(x, 0), int(row["attempt"]))
                if row["accepted"] == "True":
                    accepted[x] = accepted.get(x, 0) + 1
    return accepted, last_attempt


def _base_config(fig_dir: Path) -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        auto_random_seed=False,
        results_dir=fig_dir / "_method_results",
        output_dir=fig_dir / "_dags",
        run_hybrid_topk_gssgd_backhaul=True,
        run_hybrid_topk_location_aware_backhaul=True,
        run_wdmoe_gssgd=True,
        run_wdmoe_location_aware=True,
        run_our_similarity_aware_topk_backhaul=True,
    )


def run_figure(job: tuple[int, int, int, str, bool]) -> str:
    """Collect accepted runs for every x value of one figure (one process)."""
    number, runs, max_attempts, run_dir_text, quick = job
    figure = FIGURE_BY_NUMBER[number]
    fig_dir = _figure_dir(Path(run_dir_text), figure)
    fig_dir.mkdir(parents=True, exist_ok=True)
    obs_path = fig_dir / "observations.csv"
    att_path = fig_dir / "attempts.csv"
    # An empty file (e.g. a run interrupted before its first flush) counts as new.
    new_obs = not obs_path.exists() or obs_path.stat().st_size == 0
    new_att = not att_path.exists() or att_path.stat().st_size == 0
    accepted_counts, last_attempts = _load_progress(fig_dir)
    base = _base_config(fig_dir)
    log_path = fig_dir / "progress.log"
    values = figure.values[:2] if quick else figure.values
    with obs_path.open("a", newline="", encoding="utf-8") as obs_file, \
            att_path.open("a", newline="", encoding="utf-8") as att_file, \
            log_path.open("a", encoding="utf-8") as log:
        obs_fields = FIELDS
        if not new_obs:
            with obs_path.open(encoding="utf-8") as existing:
                header = existing.readline().strip()
            if header:
                obs_fields = tuple(header.split(","))
        obs_writer = csv.DictWriter(obs_file, fieldnames=obs_fields, extrasaction="ignore")
        att_writer = csv.DictWriter(att_file, fieldnames=ATTEMPT_FIELDS)
        if new_obs:
            obs_writer.writeheader()
        if new_att:
            att_writer.writeheader()
        accepted_attempts: dict[float, set[int]] = {}
        if att_path.exists() and not new_att:
            with att_path.open(encoding="utf-8") as existing:
                for row in csv.DictReader(existing):
                    if row["accepted"] == "True":
                        accepted_attempts.setdefault(float(row["x"]), set()).add(int(row["attempt"]))

        def one_attempt(value: float, attempt: int) -> bool:
            config = figure.apply(base, value)
            seeds = resolve_run_seeds(config, run_index=attempt - 1)
            start = time.time()
            try:
                graphs = generate_task_graphs(config, seeds, verbose=False, export_artifacts=False)
                results = run_enabled_methods(config, graphs, seeds, print_details=False, show_plot=False)
                violations = {k: len(getattr(r, "violations", []) or []) for k, r in results.items()}
                ok = len(results) == 5 and all(v == 0 for v in violations.values())
            except Exception as error:  # infeasible sample: reject it
                results, ok = {}, False
                violations = {"error": str(error)[:200]}
            if ok:
                accepted_attempts.setdefault(float(value), set()).add(attempt)
                for method_key, result in results.items():
                    obs_writer.writerow({
                        "figure": figure.number, "key": figure.key, "x": value,
                        "attempt": attempt, "accepted_index": len(accepted_attempts[float(value)]),
                        "dag_seed": seeds.dag_seed, "network_seed": seeds.network_seed,
                        "method_key": method_key, **_record(result),
                    })
            att_writer.writerow({
                "figure": figure.number, "key": figure.key, "x": value,
                "attempt": attempt, "accepted": ok,
                "seconds": round(time.time() - start, 2),
                "violations": json.dumps(violations, ensure_ascii=False),
            })
            obs_file.flush()
            att_file.flush()
            return ok

        if figure.paired:
            # Same attempt (seed) at every x; count seeds accepted at all x.
            attempt = min((last_attempts.get(float(v), 0) for v in values), default=0)
            def common_count() -> int:
                sets = [accepted_attempts.get(float(v), set()) for v in values]
                return len(set.intersection(*sets)) if sets else 0
            while common_count() < runs and attempt < max_attempts:
                attempt += 1
                for value in values:
                    if attempt <= last_attempts.get(float(value), 0):
                        continue
                    one_attempt(value, attempt)
                log.write(f"[Fig {figure.number} {figure.key}] attempt {attempt}: "
                          f"accepted at all x = {common_count()}/{runs}\n")
                log.flush()
            if common_count() < runs:
                log.write(f"WARNING: only {common_count()}/{runs} seeds accepted at every x\n")
        else:
            for value in values:
                accepted = accepted_counts.get(float(value), 0)
                attempt = last_attempts.get(float(value), 0)
                while accepted < runs and attempt < max_attempts:
                    attempt += 1
                    ok = one_attempt(value, attempt)
                    accepted += int(ok)
                    line = (f"[Fig {figure.number} {figure.key}={value}] attempt {attempt}: "
                            f"{'accepted' if ok else 'rejected'} ({accepted}/{runs})")
                    log.write(line + "\n")
                    log.flush()
                if accepted < runs:
                    log.write(f"WARNING: only {accepted}/{runs} accepted for x={value}\n")
    _plot_one(Path(run_dir_text), figure)
    return f"Fig {figure.number} done"


def _rows(fig_dir: Path) -> list[dict]:
    path = fig_dir / "observations.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Plot options (set from the command line).
PLOT_OPTIONS: dict[str, Any] = {"drop_outliers": True, "outlier_factor": 10.0, "runs": 100}


def _select_rows(figure: Figure, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Optionally drop extreme runs, then keep the first `runs` runs per x.

    A run (one attempt, all five methods) is an outlier at its x value when
    any method's plotted metric exceeds outlier_factor x that method's median
    at the same x.  The whole run is removed for every method, so all methods
    are still averaged over the same runs.  Remaining runs are taken in
    attempt order (seed order) up to `runs` per x.
    """
    factor = float(PLOT_OPTIONS["outlier_factor"])
    limit = PLOT_OPTIONS["runs"]
    kept: list[dict] = []
    removed: list[dict] = []
    good_by_x: dict[float, list[int]] = {}
    by_x: dict[float, dict[int, list[dict]]] = {}
    for row in rows:
        by_x.setdefault(float(row["x"]), {}).setdefault(int(row["attempt"]), []).append(row)
    for x, attempts in by_x.items():
        flagged: set[int] = set()
        if PLOT_OPTIONS["drop_outliers"]:
            medians = {}
            for method_key in METHOD_ORDER:
                values = [
                    float(r[figure.metric])
                    for runs in attempts.values()
                    for r in runs
                    if r["method_key"] == method_key
                ]
                if values:
                    medians[method_key] = sorted(values)[len(values) // 2]
            for attempt, runs in attempts.items():
                for r in runs:
                    median = medians.get(r["method_key"], 0.0)
                    if median > 0 and float(r[figure.metric]) > factor * median:
                        flagged.add(attempt)
                        removed.append({
                            "figure": figure.number, "x": x, "attempt": attempt,
                            "method": r["method_key"], "metric": figure.metric,
                            "value": float(r[figure.metric]), "median": median,
                        })
        good_by_x[x] = [a for a in sorted(attempts) if a not in flagged]
    if figure.paired and good_by_x:
        common = set.intersection(*(set(good) for good in good_by_x.values()))
        good_by_x = {x: [a for a in good if a in common] for x, good in good_by_x.items()}
    for x, good in good_by_x.items():
        if limit is not None:
            good = good[: int(limit)]
        for attempt in good:
            kept.extend(by_x[x][attempt])
    return kept, removed


def _plot_one(run_dir: Path, figure: Figure) -> list[dict]:
    """Draw one line chart (mean of accepted runs per x) and return summary rows."""
    source = FIGURE_BY_NUMBER[figure.source_figure_number] if figure.source_figure_number else figure
    rows = _rows(_figure_dir(run_dir, source))
    if not rows:
        return []
    rows, removed = _select_rows(figure, rows)
    PLOT_OPTIONS.setdefault("removed", []).extend(removed)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    summary = []
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for method_key in METHOD_ORDER:
        xs, ys = [], []
        for value in figure.values:
            samples = [
                float(row[figure.metric])
                for row in rows
                if row["method_key"] == method_key and float(row["x"]) == float(value)
            ]
            if not samples:
                continue
            xs.append(value)
            ys.append(fmean(samples))
            summary.append({
                "figure": figure.number, "x": value, "method": method_key,
                "metric": figure.metric, "mean": fmean(samples),
                "std": pstdev(samples) if len(samples) > 1 else 0.0, "n": len(samples),
            })
        if xs:
            color, marker, line = METHOD_STYLE[method_key]
            axis.plot(xs, ys, color=color, marker=marker, linestyle=line, linewidth=2,
                      markersize=9, label=PLOT_LABELS.get(method_key, METHOD_DISPLAY_NAMES.get(method_key, method_key)))
    if figure.log_y:
        axis.set_yscale("log")
    axis.set_xlabel(figure.x_label)
    ylabel_text = {
        "Communication cost (bandwidth + forwarding)":
            "Communication cost\n(bandwidth + forwarding)",
        "Number of groups $-$ number of single-IoT groups":
            "Number of groups $-$\nnumber of single-IoT groups",
    }.get(figure.y_label, figure.y_label)
    axis.set_ylabel(ylabel_text)
    axis.set_xticks(list(figure.values))
    axis.grid(True, linestyle="--", color="gray", alpha=0.5)
    axis.tick_params(direction="out", length=4, width=1.2)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                handlelength=2.6, columnspacing=1.2, handletextpad=0.4, borderaxespad=0.2)
    # Keep vertical titles within the axes so they clear the legend and x-axis label.
    fig.tight_layout()
    fig.canvas.draw()
    ylabel = axis.yaxis.label
    while (ylabel.get_window_extent(renderer=fig.canvas.get_renderer()).height > axis.bbox.height * 0.9
           and ylabel.get_fontsize() > 10):
        ylabel.set_fontsize(ylabel.get_fontsize() - 1)
        fig.tight_layout()
        fig.canvas.draw()
    fig.savefig(figure_dir / f"fig{figure.number:02d}_{figure.key}.png", dpi=200)
    plt.close(fig)
    return summary


def plot(run_dir: Path, figures: list[Figure]) -> None:
    summary = []
    PLOT_OPTIONS["removed"] = []
    for figure in figures:
        summary.extend(_plot_one(run_dir, figure))
    if PLOT_OPTIONS["drop_outliers"]:
        with (run_dir / "removed_outliers.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("figure", "x", "attempt", "method", "metric", "value", "median"))
            writer.writeheader()
            writer.writerows(PLOT_OPTIONS["removed"])
        removed_runs = {(r["figure"], r["x"], r["attempt"]) for r in PLOT_OPTIONS["removed"]}
        print(f"Removed {len(removed_runs)} outlier runs (> {PLOT_OPTIONS['outlier_factor']:g} x median); "
              f"list: {run_dir / 'removed_outliers.csv'}")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("figure", "x", "method", "metric", "mean", "std", "n"))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Figures: {run_dir / 'figures'}")


def _latest_unfinished(root: Path) -> Path | None:
    if not root.exists():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        if candidate.is_dir() and (candidate / "settings.json").exists() and not (candidate / "DONE").exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30, help="accepted runs per x value (default 30)")
    parser.add_argument("--max-attempts", type=int, default=5000, help="attempt limit per x value")
    parser.add_argument("--figures", default="all", help="e.g. 1,2,5 (default: all)")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel processes (default: CPU cores - 1, at most 10)")
    parser.add_argument("--quick", action="store_true", help="only the first two x values")
    parser.add_argument("--new", action="store_true", help="start a new run instead of resuming")
    parser.add_argument("--resume", type=Path, default=None, help="continue this run directory")
    parser.add_argument("--plot-only", type=Path, default=None, help="redraw figures from a run directory")
    parser.add_argument("--keep-outliers", action="store_true",
                        help="when plotting, keep all runs (by default runs where any method exceeds "
                             "--outlier-factor x its median are dropped)")
    parser.add_argument("--outlier-factor", type=float, default=10.0, help="outlier threshold (default 10 x median)")
    parser.add_argument("--plot-runs", type=int, default=10,
                        help="when plotting, average the first N non-outlier runs per x value (default 10; 0 = all)")
    args = parser.parse_args()
    PLOT_OPTIONS.update(
        drop_outliers=not args.keep_outliers,
        outlier_factor=args.outlier_factor,
        runs=(args.plot_runs or None),
    )
    figures = list(FIGURES) if args.figures == "all" else [
        FIGURE_BY_NUMBER[int(item)] for item in args.figures.split(",")
    ]
    if args.plot_only:
        plot(args.plot_only, figures)
        return

    run_dir = args.resume or (None if args.new else _latest_unfinished(OUTPUT_ROOT))
    if run_dir is None:
        run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"New run: {run_dir}")
    else:
        print(f"Resuming: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps({
        "runs_per_point": args.runs,
        "max_attempts_per_point": args.max_attempts,
        "figures": [figure.number for figure in figures],
        "acceptance": "all five methods have zero violations",
        "seeds": "dag_seed = 42 + attempt-1, network_seed = 1 + attempt-1",
        "power_allocation": "overlap",
        "fig3": "nested servers (base 10), paired seeds",
        "fig4_deadline": "average 2,4,6,8,10 s; U[avg-0.5,avg+0.5] s; min_rate=0.4e6; unchanged cost weights; communication cost; paired seeds",
        "fig7": "nested storage (base 10-14), 12 gating candidates, paired seeds",
        "fig12": "RSMA group counts from fig2 observations; no additional simulation",
        "fig13": "bandwidth costs from fig4 observations; no additional simulation",
    }, indent=2), encoding="utf-8")

    workers = args.workers or max(1, min(10, (os.cpu_count() or 2) - 1))
    simulation_numbers = list(dict.fromkeys(
        figure.source_figure_number or figure.number for figure in figures
    ))
    workers = min(workers, len(simulation_numbers))
    jobs = [(number, args.runs, args.max_attempts, str(run_dir), args.quick)
            for number in simulation_numbers]
    print(f"{len(figures)} figures from {len(jobs)} simulation sweeps, "
          f"{workers} parallel processes, {args.runs} accepted runs per point.")
    print(f"Progress of each figure: {run_dir}/figXX_*/progress.log")
    started = time.time()
    if workers == 1:
        for job in jobs:
            print(run_figure(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_figure, job): job[0] for job in jobs}
            for future in as_completed(futures):
                print(f"{future.result()}  ({(time.time() - started) / 60:.1f} min)", flush=True)
    # Redraw every figure that has data in this run directory (keeps summary.csv complete).
    plot(run_dir, list(FIGURES))
    if not args.quick:
        (run_dir / "DONE").write_text("finished\n", encoding="utf-8")
    print(f"All done in {(time.time() - started) / 3600:.2f} h. Results: {run_dir}")


if __name__ == "__main__":
    main()
