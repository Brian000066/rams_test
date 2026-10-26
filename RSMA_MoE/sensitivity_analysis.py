from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt

from config import ExperimentConfig
from experiment_runner import (
    AverageResult,
    METHOD_DISPLAY_NAMES,
    average_run_results,
    generate_task_graphs,
    print_cost_line,
    resolve_run_seeds,
    run_enabled_methods,
    rounded_cost,
    sanitize_filename,
    save_figure,
)


SWEEPS: dict[str, list[int]] = {
    "dag_num_nodes": [6, 8, 10, 12, 14],
    "topk_k": [4, 5, 6, 7, 8],
}
METHOD_ORDER = [
    "hybrid_topk_gssgd_backhaul",
    "hybrid_topk_location_aware_backhaul",
    "wdmoe_gssgd",
    "wdmoe_location_aware",
    "our_similarity_aware_topk_backhaul",
]

METHOD_COLORS = {
    "hybrid_topk_gssgd_backhaul": "#4C78A8",
    "hybrid_topk_location_aware_backhaul": "#F58518",
    "wdmoe_gssgd": "#54A24B",
    "wdmoe_location_aware": "#B279A2",
    "our_similarity_aware_topk_backhaul": "#E45756",
}


def evaluate_config(config: ExperimentConfig) -> dict[str, AverageResult]:
    run_results: list[dict[str, Any]] = []
    for run_index in range(config.num_runs):
        seeds = resolve_run_seeds(config, run_index=run_index)
        graphs = generate_task_graphs(config, seeds, verbose=False, export_artifacts=False)
        results = run_enabled_methods(
            config,
            graphs,
            seeds,
            print_details=False,
            show_plot=False,
        )
        run_results.append(results)
    return average_run_results(run_results, config)


def apply_sweep_value(base_config: ExperimentConfig, parameter_name: str, value: int) -> ExperimentConfig:
    if parameter_name == "dag_num_nodes":
        return replace(
            base_config,
            common_dag_spec=replace(base_config.common_dag_spec, num_nodes=value),
        )
    if parameter_name == "dag_num_branches":
        spec = base_config.common_dag_spec
        return replace(
            base_config,
            common_dag_spec=replace(
                spec,
                num_branches=value,
                num_nodes=max(spec.num_nodes, value + 1),
            ),
        )
    if parameter_name == "deadline_seconds":
        return replace(base_config, task_deadline_seconds_range=(value, value))
    if parameter_name == "features_per_node":
        return replace(base_config, iot_features_per_node_range=(value, value))
    return replace(base_config, **{parameter_name: value})


def run_parameter_sweep(
    base_config: ExperimentConfig,
    parameter_name: str,
    values: list[int],
) -> dict[int, dict[str, AverageResult]]:
    print(f"\nSweep: {parameter_name}")
    print("=" * 72)
    sweep_results: dict[int, dict[str, AverageResult]] = {}
    for value in values:
        config = apply_sweep_value(base_config, parameter_name, value)
        averages = evaluate_config(config)
        sweep_results[value] = averages
        print_point_summary(parameter_name, value, averages, config)
    return sweep_results

def print_point_summary(
    parameter_name: str,
    value: int,
    averages: dict[str, AverageResult],
    config: ExperimentConfig,
) -> None:
    print(f"\n{parameter_name} = {value}")
    for method_key in METHOD_ORDER:
        if method_key not in averages:
            continue
        result = averages[method_key]
        method_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        print(
            f"  {method_name}: total={result.total_cost:.2f} ({rounded_cost(result.total_cost)}), "
            f"violations={result.violation_count:.2f}"
        )
        print_cost_line("Activation", result.activation_cost, config.c_act, "activated expert memory")
        print_cost_line("Bandwidth", result.bandwidth_cost, config.c_bw, "Hz")
        print_cost_line("Forwarding", result.forwarding_cost, config.c_fwd, "forwarding events")
        print_cost_line(
            "Inference",
            result.inference_cost,
            config.c_inf,
            "executed expert complexity (MFLOP-equivalent)",
        )


def loss_bound_text(config: ExperimentConfig) -> str:
    if config.loss_threshold is None:
        return f"calibration q_hat, alpha={config.calibration_alpha}"
    return f"fixed q_hat={config.loss_threshold}"


def important_fixed_parameters(config: ExperimentConfig, swept_parameter: str, runs: int) -> list[tuple[str, Any]]:
    values: dict[str, tuple[str, Any]] = {
        "runs": ("runs/point", runs),
        "num_dags": ("graphs", config.num_dags),
        "dag_num_nodes": ("DAG nodes", config.common_dag_spec.num_nodes),
        "dag_num_branches": ("DAG branches", config.common_dag_spec.num_branches),
        "deadline_seconds": ("deadline", config.task_deadline_seconds_range),
        "features_per_node": ("features/node", config.iot_features_per_node_range),
        "num_iot_devices": ("IoT devices", config.num_iot_devices),
        "topk_k": ("Top-K", config.topk_k),
        "num_edge_servers": ("servers", config.num_edge_servers),
        "cell_radius": ("cell radius", config.cell_radius),
        "max_group_size": ("max group size", config.max_group_size),
        "loss_model": (
            "loss",
            f"{loss_bound_text(config)}, lambda={config.lambda_reconstruction}, sigma={config.reconstruction_sigma}",
        ),
        "unit_cost": ("unit cost", f"bw={config.c_bw}, act={config.c_act}, fwd={config.c_fwd}"),
    }

    per_sweep_context: dict[str, list[str]] = {
        "dag_num_nodes": [
            "runs", "num_dags", "dag_num_branches", "deadline_seconds",
            "features_per_node", "num_iot_devices", "topk_k",
            "num_edge_servers", "loss_model", "unit_cost",
        ],
        "dag_num_branches": [
            "runs", "num_dags", "dag_num_nodes", "deadline_seconds",
            "features_per_node", "num_iot_devices", "topk_k",
            "num_edge_servers", "loss_model", "unit_cost",
        ],
        "num_dags": [
            "runs", "dag_num_nodes", "dag_num_branches", "deadline_seconds",
            "features_per_node", "num_iot_devices", "topk_k",
            "num_edge_servers", "loss_model", "unit_cost",
        ],
        "deadline_seconds": [
            "runs", "num_dags", "dag_num_nodes", "dag_num_branches",
            "features_per_node", "num_iot_devices", "topk_k",
            "num_edge_servers", "loss_model", "unit_cost",
        ],
        "features_per_node": [
            "runs", "num_dags", "dag_num_nodes", "dag_num_branches",
            "deadline_seconds", "num_iot_devices", "topk_k",
            "num_edge_servers", "max_group_size", "loss_model", "unit_cost",
        ],
        "num_iot_devices": [
            "runs", "num_dags", "dag_num_nodes", "dag_num_branches",
            "deadline_seconds", "features_per_node", "topk_k",
            "num_edge_servers", "cell_radius", "max_group_size",
            "loss_model", "unit_cost",
        ],
        "topk_k": [
            "runs", "num_dags", "dag_num_nodes", "dag_num_branches",
            "deadline_seconds", "features_per_node", "num_iot_devices",
            "num_edge_servers", "max_group_size", "loss_model", "unit_cost",
        ],
    }
    keys = per_sweep_context.get(swept_parameter, list(values))
    return [values[key] for key in keys if key in values and key != swept_parameter]


def fixed_parameter_text(config: ExperimentConfig, swept_parameter: str, runs: int) -> str:
    lines = ["Fixed parameters"]
    for name, value in important_fixed_parameters(config, swept_parameter, runs):
        lines.append(f"{name}: {value}")
    return "\n".join(lines)


def plot_sweep_metric(
    parameter_name: str,
    sweep_results: dict[int, dict[str, AverageResult]],
    base_config: ExperimentConfig,
    runs: int,
    metric_name: str,
    y_label: str,
    title: str,
) -> None:
    values = sorted(sweep_results)
    fig, axis = plt.subplots(figsize=(12, 5.8))
    for method_key in METHOD_ORDER:
        metric_values = [
            getattr(sweep_results[value][method_key], metric_name)
            for value in values
            if method_key in sweep_results[value]
        ]
        if len(metric_values) != len(values):
            continue
        axis.plot(
            values,
            metric_values,
            marker="o",
            linewidth=2.2,
            label=METHOD_DISPLAY_NAMES.get(method_key, method_key),
            color=METHOD_COLORS.get(method_key),
        )

    axis.set_title(title)
    x_axis_labels = {
        "dag_num_nodes": "Number of graph nodes",
        "topk_k": "Number of K",
    }
    axis.set_xlabel(x_axis_labels.get(parameter_name, parameter_name))
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    fig.subplots_adjust(right=0.68)
    fig.text(
        0.71,
        0.5,
        fixed_parameter_text(base_config, parameter_name, runs),
        va="center",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
    )
    saved_path = save_figure(
        fig,
        base_config.results_dir / "figure",
        f"{sanitize_filename(title)}.png",
    )
    print(f"Saved figure: {saved_path.resolve()}")


def plot_total_cost_sweep(
    parameter_name: str,
    sweep_results: dict[int, dict[str, AverageResult]],
    base_config: ExperimentConfig,
    runs: int,
) -> None:
    total_cost_titles = {
        "dag_num_nodes": "Average Total Cost vs Number of Graph Nodes",
        "topk_k": "Average Total Cost vs K",
    }
    plot_sweep_metric(
        parameter_name=parameter_name,
        sweep_results=sweep_results,
        base_config=base_config,
        runs=runs,
        metric_name="total_cost",
        y_label="Average Total Cost",
        title=total_cost_titles.get(parameter_name, f"Average Total Cost vs {parameter_name}"),
    )


def plot_extra_sweep_metric(
    parameter_name: str,
    sweep_results: dict[int, dict[str, AverageResult]],
    base_config: ExperimentConfig,
    runs: int,
) -> None:
    # Each physical hop has its own edge weight. forwarding_usage is therefore
    # the accumulated minimum-route weight, rather than an unweighted hop count.
    extra_plot_specs = {
        "dag_num_nodes": [
            (
                "forwarding_usage",
                "Average Weighted Route Cost",
                "Average Weighted Route Cost vs Number of Graph Nodes",
            ),
        ],
        "topk_k": [
            ("inference_time_ms", "Average Inference Time (ms)", "Average Inference Time vs K"),
        ],
    }
    for metric_name, y_label, title in extra_plot_specs.get(parameter_name, []):
        plot_sweep_metric(
            parameter_name=parameter_name,
            sweep_results=sweep_results,
            base_config=base_config,
            runs=runs,
            metric_name=metric_name,
            y_label=y_label,
            title=title,
        )


def plot_sweep(
    parameter_name: str,
    sweep_results: dict[int, dict[str, AverageResult]],
    base_config: ExperimentConfig,
    runs: int,
) -> None:
    """Plot both requested metrics for one sweep (backward-compatible helper)."""
    plot_total_cost_sweep(parameter_name, sweep_results, base_config, runs)
    plot_extra_sweep_metric(parameter_name, sweep_results, base_config, runs)


def run_all_sweeps(
    runs: int,
    base_config: ExperimentConfig | None = None,
) -> dict[str, dict[int, dict[str, AverageResult]]]:
    base_config = replace(
        base_config or ExperimentConfig(),
        auto_random_seed=False,
        num_runs=runs,
    )
    all_results: dict[str, dict[int, dict[str, AverageResult]]] = {}
    for parameter_name, values in SWEEPS.items():
        all_results[parameter_name] = run_parameter_sweep(base_config, parameter_name, values)

    # Follow the requested table order: both total-cost figures first, followed
    # by weighted route cost and inference time.
    for parameter_name in SWEEPS:
        plot_total_cost_sweep(parameter_name, all_results[parameter_name], base_config, runs)
    for parameter_name in SWEEPS:
        plot_extra_sweep_metric(parameter_name, all_results[parameter_name], base_config, runs)
    plt.show()
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task and network sensitivity plots for RSMA-MoE methods.")
    parser.add_argument("--runs", type=int, default=100, help="Number of repeated runs per x-axis point.")
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    run_all_sweeps(args.runs)


if __name__ == "__main__":
    main()


