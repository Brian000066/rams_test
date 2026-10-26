"""Generate the twelve requested RSMA-MoE sensitivity figures.

The script changes one independent variable at a time, executes every enabled
method with the same DAG/network seeds, averages the requested metrics, and
writes PNG figures plus the raw observations used to draw them.

Examples
--------
Quick code-path check with JRGSP and two points per sweep::

    python3 run_requested_sensitivity_figures.py --quick --runs 1 \
        --max-attempts-per-point 20 --methods jrgep

Full five-method experiment with 100 repetitions per point::

    python3 run_requested_sensitivity_figures.py --runs 100 --methods all
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import ExperimentConfig
from experiment_runner import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    generate_task_graphs,
    resolve_run_seeds,
    run_enabled_methods,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "requested_sensitivity"

METHOD_ORDER = (
    "hybrid_topk_gssgd_backhaul",
    "hybrid_topk_location_aware_backhaul",
    "wdmoe_gssgd",
    "wdmoe_location_aware",
    "our_similarity_aware_topk_backhaul",
)
METHOD_ALIASES = {
    "hybrid_topk_gssgd": "hybrid_topk_gssgd_backhaul",
    "hybrid_topk_location_aware": "hybrid_topk_location_aware_backhaul",
    "wdmoe_gssgd": "wdmoe_gssgd",
    "wdmoe_location_aware": "wdmoe_location_aware",
    "jrgep": "our_similarity_aware_topk_backhaul",
    "jrgsp": "our_similarity_aware_topk_backhaul",
}
METHOD_COLORS = {
    "hybrid_topk_gssgd_backhaul": "#4C78A8",
    "hybrid_topk_location_aware_backhaul": "#F58518",
    "wdmoe_gssgd": "#54A24B",
    "wdmoe_location_aware": "#B279A2",
    "our_similarity_aware_topk_backhaul": "#E45756",
}
METHOD_LINESTYLES = {
    "hybrid_topk_gssgd_backhaul": "-",
    "hybrid_topk_location_aware_backhaul": "--",
    "wdmoe_gssgd": "-.",
    "wdmoe_location_aware": ":",
    "our_similarity_aware_topk_backhaul": (0, (5, 1, 1, 1)),
}
METHOD_MARKERS = {
    "hybrid_topk_gssgd_backhaul": "o",
    "hybrid_topk_location_aware_backhaul": "s",
    "wdmoe_gssgd": "^",
    "wdmoe_location_aware": "D",
    "our_similarity_aware_topk_backhaul": "X",
}

# Each sweep is evaluated only once. Multiple requested figures can reuse the
# same observations, which avoids rerunning an identical experiment.
SWEEP_VALUES: dict[str, tuple[float | int, ...]] = {
    "shared_subtask_ratio": (0.0, 0.2, 0.4, 0.6, 0.8),
    "num_iot_devices": (50, 75, 100, 125, 150),
    "num_iot_features": (40, 60, 80, 100, 120),
    "time_seconds": (1, 2, 3, 4, 5),
    "max_group_size": (1, 2, 3, 4, 5),
    "features_per_subtask": (1, 2, 3, 4, 5),
    # The configured model table contains eight heterogeneous experts. A
    # global-expert sweep therefore uses valid prefixes of that same table.
    "num_experts": (4, 5, 6, 7, 8),
}


@dataclass(frozen=True)
class FigureSpec:
    number: int
    sweep: str
    metric: str
    filename: str
    title: str
    x_label: str
    y_label: str


FIGURE_SPECS = (
    FigureSpec(
        1,
        "shared_subtask_ratio",
        "forwarding_cost",
        "01_shared_task_node_ratio_vs_forwarding_cost.png",
        "Forwarding Cost vs Shared Task-Node Ratio",
        "Shared task nodes / total task nodes",
        "Average forwarding cost",
    ),
    FigureSpec(
        2,
        "shared_subtask_ratio",
        "activation_cost",
        "02_shared_task_node_ratio_vs_activation_cost.png",
        "Activation Cost vs Shared Task-Node Ratio",
        "Shared task nodes / total task nodes",
        "Average activation cost",
    ),
    FigureSpec(
        3,
        "num_iot_devices",
        "selected_iot_count",
        "03_iot_count_vs_selected_iot_count.png",
        "Selected IoTs vs Number of IoT Devices",
        "Number of IoT devices",
        "Average number of selected IoTs",
    ),
    FigureSpec(
        4,
        "num_iot_devices",
        "total_cost",
        "04_iot_count_vs_total_cost.png",
        "Total Cost vs Number of IoT Devices",
        "Number of IoT devices",
        "Average total cost",
    ),
    FigureSpec(
        5,
        "num_iot_features",
        "bandwidth_cost",
        "05_iot_feature_count_vs_bandwidth_cost.png",
        "Bandwidth Cost vs Number of IoT Features",
        "Number of global IoT features",
        "Average bandwidth cost",
    ),
    FigureSpec(
        6,
        "num_iot_features",
        "iot_group_count",
        "06_iot_feature_count_vs_iot_group_count.png",
        "IoT Group Count vs Number of IoT Features",
        "Number of global IoT features",
        "Average number of IoT groups",
    ),
    FigureSpec(
        7,
        "time_seconds",
        "completed_subtask_ratio",
        "07_time_vs_completed_subtask_ratio.png",
        "On-Time Subtask Completion Ratio vs Time",
        "Elapsed time (seconds)",
        "Average on-time completed subtask ratio",
    ),
    FigureSpec(
        8,
        "max_group_size",
        "bandwidth_cost",
        "08_group_size_vs_bandwidth_cost.png",
        "Bandwidth Cost vs Maximum IoT Group Size",
        "Maximum IoT group size",
        "Average bandwidth cost",
    ),
    FigureSpec(
        9,
        "features_per_subtask",
        "selected_iot_count",
        "09_required_features_vs_selected_iot_count.png",
        "Selected IoTs vs Required Features per Subtask",
        "Average required features per subtask",
        "Average number of selected IoTs",
    ),
    FigureSpec(
        10,
        "num_experts",
        "total_cost",
        "10_global_expert_count_vs_total_cost.png",
        "Total Cost vs Number of Global Experts",
        "Number of global experts",
        "Average total cost",
    ),
    FigureSpec(
        11,
        "num_experts",
        "activation_cost",
        "11_global_expert_count_vs_activation_cost.png",
        "Activation Cost vs Number of Global Experts",
        "Number of global experts",
        "Average activation cost",
    ),
    FigureSpec(
        12,
        "shared_subtask_ratio",
        "expert_reuse_ratio",
        "12_shared_task_node_ratio_vs_expert_reuse_ratio.png",
        "Expert Reuse Ratio vs Shared Task-Node Ratio",
        "Shared task nodes / total task nodes",
        "Expert reuse ratio",
    ),
)


def _method_only_config(config: ExperimentConfig, method_key: str) -> ExperimentConfig:
    """Return a config that executes exactly one method."""
    return replace(
        config,
        run_hybrid_topk_gssgd_backhaul=(method_key == "hybrid_topk_gssgd_backhaul"),
        run_hybrid_topk_location_aware_backhaul=(
            method_key == "hybrid_topk_location_aware_backhaul"
        ),
        run_wdmoe_gssgd=(method_key == "wdmoe_gssgd"),
        run_wdmoe_location_aware=(method_key == "wdmoe_location_aware"),
        run_our_similarity_aware_topk_backhaul=(
            method_key == "our_similarity_aware_topk_backhaul"
        ),
    )


def _apply_sweep_value(
    base: ExperimentConfig,
    sweep: str,
    value: float | int,
) -> ExperimentConfig:
    if sweep == "shared_subtask_ratio":
        return replace(base, shared_subtask_ratio=float(value))
    if sweep == "num_iot_devices":
        return replace(base, num_iot_devices=int(value))
    if sweep == "num_iot_features":
        return replace(base, num_iot_features=int(value))
    if sweep == "time_seconds":
        # Figure 7 observes progress under the configured per-task deadlines;
        # it does not change them.  Deterministic per-run seeds ensure that the
        # 1--5 second points are evaluated on the same DAG/network samples.
        return replace(base, auto_random_seed=False)
    if sweep == "max_group_size":
        return replace(base, max_group_size=int(value))
    if sweep == "features_per_subtask":
        feature_count = int(value)
        return replace(
            base,
            iot_features_per_node_range=(feature_count, feature_count),
        )
    if sweep == "num_experts":
        expert_count = int(value)
        if not 1 <= expert_count <= len(base.expert_hidden_dimensions):
            raise ValueError(
                "num_experts sweep must fit the configured expert model table"
            )
        hidden = base.expert_hidden_dimensions[:expert_count]
        parameters = base.expert_parameter_counts[:expert_count]
        memory = base.expert_memory_sizes_mb[:expert_count]
        inference = base.expert_inference_times_ms[:expert_count]
        return replace(
            base,
            num_experts=expert_count,
            expert_hidden_dimensions=hidden,
            expert_parameter_counts=parameters,
            expert_memory_sizes_mb=memory,
            expert_memory_range=(min(memory), max(memory)),
            expert_inference_times_ms=inference,
            topk_k=min(base.topk_k, expert_count),
            experts_per_server=min(base.experts_per_server, expert_count),
        )
    raise KeyError(f"Unknown sweep: {sweep}")


def _actual_shared_ratio(graphs: Sequence[Any]) -> float:
    shared_count = sum(
        int(graph.graph.get("shared_subtask_instance_count", 0))
        for graph in graphs
    )
    total_count = sum(int(graph.number_of_nodes()) for graph in graphs)
    return shared_count / total_count if total_count else 0.0


def _actual_features_per_subtask(graphs: Sequence[Any]) -> float:
    feature_counts = [
        len(graph.nodes[node].get("required_data", {}).get("iot_features", ()))
        for graph in graphs
        for node in graph.nodes
    ]
    return fmean(feature_counts) if feature_counts else 0.0


def _axis_value(
    sweep: str,
    requested_value: float | int,
    graphs: Sequence[Any],
) -> float:
    if sweep == "shared_subtask_ratio":
        return _actual_shared_ratio(graphs)
    if sweep == "features_per_subtask":
        return _actual_features_per_subtask(graphs)
    return float(requested_value)


def _selected_iot_occurrence_count(result: Any) -> int:
    """Count IoT selections with repetition across all RSMA groups.

    If one physical IoT is selected into two different groups, it contributes
    two selections.  This measures group-membership/selection load rather than
    the number of distinct physical IoT devices used at least once.
    """
    return sum(
        1
        for groups in (getattr(result, "rsma_groups", {}) or {}).values()
        for group in groups
        for _device_id in group
    )


def _iot_group_count(result: Any) -> int:
    return sum(
        len(groups)
        for groups in (getattr(result, "rsma_groups", {}) or {}).values()
    )


def _expert_reuse_ratio(result: Any) -> float:
    """Assignments after the first use of a server/expert pair are reuse.

    This implements the requested definition:
    reused expert assignments / all expert assignments.
    """
    assignments = [
        (str(server_id), str(expert_id))
        for pairs in (getattr(result, "subtask_assignment", {}) or {}).values()
        for server_id, expert_id in pairs
    ]
    if not assignments:
        return 0.0
    reused = len(assignments) - len(set(assignments))
    return reused / len(assignments)


def _completed_subtask_ratio(
    result: Any,
    graphs: Sequence[Any],
    time_limit_seconds: float,
    time_tolerance: float = 1e-3,
) -> float:
    """Return the fraction of subtask instances finished by a time limit.

    Shared subtasks are counted once per task/DAG occurrence because each DAG
    node is an independently scheduled subtask instance.  A missing finish
    time is treated as unfinished.
    """
    deadlines_by_key = {
        (
            str(
                graph.graph.get(
                    "task_id",
                    f"task_{graph.graph.get('graph_index', index)}",
                )
            ),
            str(node_id),
        ): float(graph.graph.get("deadline_seconds", float("inf")))
        for index, graph in enumerate(graphs, start=1)
        for node_id in graph.nodes
    }
    if not deadlines_by_key:
        return 0.0
    evaluation = getattr(result, "evaluation", None)
    timing = getattr(evaluation, "timing", None)
    finish_times = getattr(timing, "subtask_finish_global", {}) or {}
    completed_count = sum(
        1
        for key, deadline_seconds in deadlines_by_key.items()
        if float(finish_times.get(key, float("inf")))
        <= min(float(time_limit_seconds), deadline_seconds) + time_tolerance
    )
    return completed_count / len(deadlines_by_key)


def _blocking_violation_count(result: Any, sweep: str) -> int:
    """Return violations that should reject an experimental attempt.

    For the time/completion curve, a deadline miss is an outcome represented
    by the y-axis rather than a reason to discard the observation.  All other
    constraint violations remain blocking.
    """
    violations = getattr(result, "violations", ()) or ()
    if sweep != "time_seconds":
        return len(violations)
    return sum(
        1
        for violation in violations
        if not str(violation).startswith("C1 deadline:")
    )


def _observation(
    *,
    sweep: str,
    requested_x: float | int,
    actual_x: float,
    accepted_index: int,
    attempt_number: int,
    method_key: str,
    result: Any,
    graphs: Sequence[Any],
    dag_seed: int,
    network_seed: int,
) -> dict[str, Any]:
    return {
        "sweep": sweep,
        "requested_x": float(requested_x),
        "actual_x": float(actual_x),
        "run_index": accepted_index - 1,
        "accepted_index": accepted_index,
        "attempt": attempt_number,
        "dag_seed": dag_seed,
        "network_seed": network_seed,
        "method_key": method_key,
        "method_name": METHOD_DISPLAY_NAMES.get(method_key, method_key),
        "total_cost": float(getattr(result, "total_cost", 0.0)),
        "activation_cost": float(getattr(result, "activation_cost", 0.0)),
        "bandwidth_cost": float(getattr(result, "bandwidth_cost", 0.0)),
        "forwarding_cost": float(getattr(result, "forwarding_cost", 0.0)),
        "selected_iot_count": _selected_iot_occurrence_count(result),
        "iot_group_count": _iot_group_count(result),
        "completed_subtask_ratio": _completed_subtask_ratio(
            result,
            graphs,
            time_limit_seconds=float(requested_x),
        ),
        "expert_reuse_ratio": _expert_reuse_ratio(result),
        "violation_count": len(getattr(result, "violations", ()) or ()),
        "blocking_violation_count": _blocking_violation_count(result, sweep),
    }


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return fmean(finite) if finite else float("nan")


def _series_for_figure(
    observations: Sequence[dict[str, Any]],
    spec: FigureSpec,
    method_key: str,
) -> tuple[list[float], list[float]]:
    matching = [
        row
        for row in observations
        if row["sweep"] == spec.sweep and row["method_key"] == method_key
    ]
    requested_values = sorted({float(row["requested_x"]) for row in matching})
    xs: list[float] = []
    ys: list[float] = []
    for requested_value in requested_values:
        rows = [
            row
            for row in matching
            if float(row["requested_x"]) == requested_value
        ]
        x_value = _mean(row["actual_x"] for row in rows)
        y_value = _mean(row[spec.metric] for row in rows)
        if math.isfinite(x_value) and math.isfinite(y_value):
            xs.append(x_value)
            ys.append(y_value)
    return xs, ys


def _plot_figure(
    observations: Sequence[dict[str, Any]],
    spec: FigureSpec,
    method_keys: Sequence[str],
    figure_dir: Path,
    dpi: int,
) -> Path:
    fig, axis = plt.subplots(figsize=(8.6, 5.4))
    plotted = False
    series_by_method = {
        method_key: _series_for_figure(observations, spec, method_key)
        for method_key in method_keys
    }
    for method_key in method_keys:
        xs, ys = series_by_method[method_key]
        if not xs:
            continue
        plotted = True
        axis.plot(
            xs,
            ys,
            marker=METHOD_MARKERS[method_key],
            linestyle=METHOD_LINESTYLES[method_key],
            linewidth=2.0,
            markersize=5.5,
            color=METHOD_COLORS[method_key],
            label=METHOD_DISPLAY_NAMES.get(method_key, method_key),
        )

    axis.set_title(spec.title)
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel(spec.y_label)
    axis.grid(True, linestyle="--", alpha=0.3)
    if spec.metric in {"completed_subtask_ratio", "expert_reuse_ratio"}:
        axis.set_ylim(-0.02, 1.02)
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "No successful observations",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    overlap_labels: list[str] = []
    visited: set[str] = set()
    for method_key in method_keys:
        if method_key in visited:
            continue
        xs, ys = series_by_method[method_key]
        if not xs:
            continue
        matching = [method_key]
        for other_key in method_keys:
            if other_key == method_key or other_key in visited:
                continue
            other_xs, other_ys = series_by_method[other_key]
            if (
                len(xs) == len(other_xs)
                and len(ys) == len(other_ys)
                and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(xs, other_xs))
                and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(ys, other_ys))
            ):
                matching.append(other_key)
        visited.update(matching)
        if len(matching) > 1:
            overlap_labels.append(
                " = ".join(METHOD_DISPLAY_NAMES.get(key, key) for key in matching)
            )

    if overlap_labels:
        fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
        fig.text(
            0.01,
            0.015,
            "Overlapping curves: " + "; ".join(overlap_labels),
            ha="left",
            va="bottom",
            fontsize=7.5,
        )
    else:
        fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    destination = figure_dir / spec.filename
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def _write_observations(path: Path, observations: Sequence[dict[str, Any]]) -> None:
    if not observations:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)


def _append_observations(
    path: Path,
    observations: Sequence[dict[str, Any]],
) -> None:
    """Persist newly accepted rows immediately during a long experiment."""
    if not observations:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(observations[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(observations)
        file.flush()


def _parse_methods(raw: str) -> tuple[str, ...]:
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names or names == ["all"]:
        return METHOD_ORDER
    resolved: list[str] = []
    for name in names:
        method_key = METHOD_ALIASES.get(name, name)
        if method_key not in METHOD_ORDER:
            choices = ", ".join(("all", *METHOD_ALIASES))
            raise ValueError(f"Unknown method {name!r}. Choices: {choices}")
        if method_key not in resolved:
            resolved.append(method_key)
    return tuple(resolved)


def _parse_sweeps(raw: str) -> tuple[str, ...]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names or names == ["all"]:
        return tuple(SWEEP_VALUES)
    unknown = [name for name in names if name not in SWEEP_VALUES]
    if unknown:
        raise ValueError(
            f"Unknown sweeps: {', '.join(unknown)}. "
            f"Choices: {', '.join(SWEEP_VALUES)}"
        )
    return tuple(dict.fromkeys(names))


def run_requested_figures(
    *,
    runs: int,
    max_attempts_per_point: int | None,
    method_keys: Sequence[str],
    sweep_names: Sequence[str],
    quick: bool,
    dag_seed: int,
    network_seed: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if (
        max_attempts_per_point is not None
        and max_attempts_per_point < runs
    ):
        raise ValueError("max_attempts_per_point must be at least runs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    figure_dir = run_dir / "figure"
    working_dir = run_dir / "_working"
    run_dir.mkdir(parents=True, exist_ok=False)

    base = replace(
        ExperimentConfig(),
        auto_random_seed=True,
        random_seed=dag_seed,
        network_random_seed=network_seed,
        num_runs=runs,
        results_dir=working_dir / "latest_method_results",
        output_dir=working_dir / "latest_generated_dags",
    )

    observations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    point_progress: dict[str, dict[str, Any]] = {}
    completed = True
    incomplete_point: str | None = None
    attempt_log_path = run_dir / "attempts.csv"
    observation_path = run_dir / "raw_observations.csv"
    attempt_fields = (
        "sweep",
        "requested_x",
        "actual_x",
        "attempt",
        "dag_seed",
        "network_seed",
        "accepted",
        "accepted_count",
        "target_accepted",
        "total_violations",
        "violation_counts",
        "blocking_violation_counts",
        "method_errors",
    )
    with attempt_log_path.open("w", newline="", encoding="utf-8") as file:
        attempt_writer = csv.DictWriter(file, fieldnames=attempt_fields)
        attempt_writer.writeheader()
        stop = False
        for sweep in sweep_names:
            configured_values = SWEEP_VALUES[sweep]
            values = (
                (configured_values[0], configured_values[-1])
                if quick and len(configured_values) > 1
                else configured_values
            )
            for value in values:
                point_config = _apply_sweep_value(base, sweep, value)
                point_key = f"{sweep}:{value}"
                accepted_count = 0
                attempt_number = 0
                while accepted_count < runs and (
                    max_attempts_per_point is None
                    or attempt_number < max_attempts_per_point
                ):
                    attempt_number += 1
                    seeds = resolve_run_seeds(
                        point_config,
                        run_index=attempt_number - 1,
                    )
                    graphs = generate_task_graphs(
                        point_config,
                        seeds,
                        verbose=False,
                        export_artifacts=False,
                    )
                    actual_x = _axis_value(sweep, value, graphs)
                    candidate_rows: list[dict[str, Any]] = []
                    violation_counts: dict[str, int] = {}
                    blocking_violation_counts: dict[str, int] = {}
                    attempt_errors: dict[str, str] = {}
                    print(
                        f"[{sweep}] x={value}, attempt="
                        f"{attempt_number}/"
                        f"{max_attempts_per_point if max_attempts_per_point is not None else '∞'}, "
                        f"accepted={accepted_count}/{runs}, "
                        f"DAG seed={seeds.dag_seed}, "
                        f"network seed={seeds.network_seed}"
                    )
                    for method_key in method_keys:
                        method_config = _method_only_config(
                            point_config,
                            method_key,
                        )
                        try:
                            method_results = run_enabled_methods(
                                method_config,
                                graphs,
                                seeds,
                                print_details=False,
                                show_plot=False,
                            )
                            result = method_results[method_key]
                        except Exception as error:
                            message = f"{type(error).__name__}: {error}"
                            attempt_errors[method_key] = message
                            errors.append(
                                {
                                    "sweep": sweep,
                                    "requested_x": float(value),
                                    "actual_x": actual_x,
                                    "attempt": attempt_number,
                                    "dag_seed": seeds.dag_seed,
                                    "network_seed": seeds.network_seed,
                                    "method_key": method_key,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                }
                            )
                            print(
                                "  failed: "
                                f"{METHOD_DISPLAY_NAMES.get(method_key, method_key)}: "
                                f"{message}"
                            )
                            continue
                        row = _observation(
                            sweep=sweep,
                            requested_x=value,
                            actual_x=actual_x,
                            accepted_index=accepted_count + 1,
                            attempt_number=attempt_number,
                            method_key=method_key,
                            result=result,
                            graphs=graphs,
                            dag_seed=seeds.dag_seed,
                            network_seed=seeds.network_seed,
                        )
                        candidate_rows.append(row)
                        violation_counts[method_key] = int(
                            row["violation_count"]
                        )
                        blocking_violation_counts[method_key] = int(
                            row["blocking_violation_count"]
                        )

                    accepted = (
                        len(candidate_rows) == len(method_keys)
                        and not attempt_errors
                        and all(
                            count == 0
                            for count in blocking_violation_counts.values()
                        )
                    )
                    if accepted:
                        accepted_count += 1
                        _append_observations(observation_path, candidate_rows)
                        observations.extend(candidate_rows)
                    total_violations = sum(violation_counts.values())
                    attempt_writer.writerow(
                        {
                            "sweep": sweep,
                            "requested_x": float(value),
                            "actual_x": actual_x,
                            "attempt": attempt_number,
                            "dag_seed": seeds.dag_seed,
                            "network_seed": seeds.network_seed,
                            "accepted": "yes" if accepted else "no",
                            "accepted_count": accepted_count,
                            "target_accepted": runs,
                            "total_violations": total_violations,
                            "violation_counts": json.dumps(
                                violation_counts,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "blocking_violation_counts": json.dumps(
                                blocking_violation_counts,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "method_errors": json.dumps(
                                attempt_errors,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        }
                    )
                    file.flush()
                    print(
                        f"  violations={total_violations}, "
                        f"accepted={'yes' if accepted else 'no'}, "
                        f"accepted_runs={accepted_count}/{runs}"
                    )

                point_complete = accepted_count == runs
                point_progress[point_key] = {
                    "sweep": sweep,
                    "requested_x": float(value),
                    "attempt_count": attempt_number,
                    "accepted_count": accepted_count,
                    "target_accepted": runs,
                    "completed": point_complete,
                }
                if not point_complete:
                    completed = False
                    incomplete_point = point_key
                    stop = True
                    break
            if stop:
                break

    _write_observations(observation_path, observations)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "settings": {
                    "runs_per_point": runs,
                    "acceptance_rule": (
                        "accept an attempt only when every selected method "
                        "has zero blocking violations; for the time sweep "
                        "only, C1 deadline misses are measured by the on-time "
                        "completed-subtask ratio and are not blocking"
                    ),
                    "max_attempts_per_point": max_attempts_per_point,
                    "methods": list(method_keys),
                    "sweeps": list(sweep_names),
                    "quick": quick,
                    "dag_seed": dag_seed,
                    "network_seed": network_seed,
                    "sweep_values": {
                        name: list(SWEEP_VALUES[name]) for name in sweep_names
                    },
                    "expert_reuse_definition": (
                        "assignments after the first occurrence of each "
                        "(server, expert) pair / all expert assignments"
                    ),
                },
                "figures": [
                    asdict(spec)
                    for spec in FIGURE_SPECS
                    if spec.sweep in sweep_names
                ],
                "successful_observation_count": len(observations),
                "completed": completed,
                "incomplete_point": incomplete_point,
                "point_progress": point_progress,
                "errors": errors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not completed:
        raise RuntimeError(
            f"Point {incomplete_point} did not reach {runs} acceptable "
            f"accepted runs within {max_attempts_per_point} attempts. "
            "No figures were generated; inspect attempts.csv and "
            "run_summary.json."
        )

    saved_figures = [
        _plot_figure(
            observations,
            spec,
            method_keys,
            figure_dir,
            base.image_dpi,
        )
        for spec in FIGURE_SPECS
        if spec.sweep in sweep_names
    ]
    print(f"Saved {len(saved_figures)} figure(s) to {figure_dir.resolve()}")
    print(f"Raw observations: {(run_dir / 'raw_observations.csv').resolve()}")
    print(f"Attempt log: {attempt_log_path.resolve()}")
    print(f"Run summary: {(run_dir / 'run_summary.json').resolve()}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the twelve requested RSMA-MoE sensitivity figures."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Constraint-acceptable runs required per x-axis point.",
    )
    parser.add_argument(
        "--max-attempts-per-point",
        type=int,
        default=0,
        help=(
            "Maximum attempts per point. Use 0 (the default) for no limit, "
            "so each point runs until --runs acceptable results are collected."
        ),
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods: all, hybrid_topk_gssgd, "
            "hybrid_topk_location_aware, wdmoe_gssgd, "
            "wdmoe_location_aware, jrgep."
        ),
    )
    parser.add_argument(
        "--sweeps",
        default="all",
        help=(
            "Comma-separated sweep names or all: "
            + ", ".join(SWEEP_VALUES)
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only the first and last x-axis values for a quick trial.",
    )
    parser.add_argument("--dag-seed", type=int, default=42)
    parser.add_argument("--network-seed", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    run_requested_figures(
        runs=args.runs,
        max_attempts_per_point=(
            None
            if args.max_attempts_per_point == 0
            else args.max_attempts_per_point
        ),
        method_keys=_parse_methods(args.methods),
        sweep_names=_parse_sweeps(args.sweeps),
        quick=args.quick,
        dag_seed=args.dag_seed,
        network_seed=args.network_seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
