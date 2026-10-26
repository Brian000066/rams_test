from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compared_method.hybrid_topk_location_aware_backhaul import (
    run_chained_pipeline as run_hybrid_topk_location_aware_backhaul,
)
from compared_method.hybrid_topk_gssgd_backhaul import run_hybrid_topk_gssgd_backhaul
from compared_method.WDMoE import run_wdmoe_location_aware, run_wdmoe_gssgd
from our_alg import run_similarity_aware_topk_backhaul
from config import ExperimentConfig
from generate_dags import build_dag_specs, generate_multiple_dags
from expert_deployment import build_expert_deployment
import rsma_integration
import expert_deployment
from task_metadata import GatingSpec
from rsma_integration import graphs_to_tasks
from network_graph_export import export_network_graph_artifacts
from utils.formulation import FormulationConfig, FormulationEvaluator, normalize_tasks


@dataclass(frozen=True)
class RunSeeds:
    dag_seed: int
    network_seed: int


@dataclass(frozen=True)
class AverageResult:
    total_cost: float
    activation_cost: float
    bandwidth_cost: float
    forwarding_cost: float
    inference_cost: float
    violation_count: float
    activation_usage: float = 0.0
    bandwidth_usage: float = 0.0
    forwarding_usage: float = 0.0
    inference_time_ms: float = 0.0
    avg_servers_per_subtask: float = 0.0
    avg_experts_per_subtask: float = 0.0
    violations: tuple[Any, ...] = ()

def rounded_cost(value: float) -> int:
    return int(value + 0.5)


def resolve_run_seeds(config: ExperimentConfig, run_index: int = 0) -> RunSeeds:
    if not config.auto_random_seed:
        return RunSeeds(
            dag_seed=config.random_seed + run_index,
            network_seed=config.network_random_seed + run_index,
        )

    rng = random.SystemRandom()
    return RunSeeds(
        dag_seed=rng.randint(1, 2_147_483_647),
        network_seed=rng.randint(1, 2_147_483_647),
    )


def build_gating_spec(config: ExperimentConfig, seeds: RunSeeds) -> GatingSpec:
    """Gating settings plus the expert deployment that the schedulers will use.

    The deployment is rebuilt from seeds.network_seed exactly as in
    rsma_integration.build_simple_servers, so each subtask's anchored top
    experts are stored on a real server in every method.
    """
    # Nested server design: tasks (gating anchors) only see the base servers,
    # so the same seed gives identical tasks for every server count.
    deployment_servers = (
        min(config.num_edge_servers, config.nested_server_base)
        if config.nested_server_base is not None
        else config.num_edge_servers
    )
    # Nested storage design: anchors come from the base deployment, which is
    # contained in every larger one, so tasks are identical for every average.
    expert_deployment.NESTED_STORAGE_BASE = config.nested_storage_base
    deployment = build_expert_deployment(
        num_experts=config.num_experts,
        num_servers=deployment_servers,
        experts_per_server=(
            config.nested_storage_base
            if config.nested_storage_base is not None
            else config.experts_per_server
        ),
        seed=seeds.network_seed,
    )
    return GatingSpec(
        num_candidates=config.gating_num_candidates,
        anchor_count=config.gating_anchor_count,
        secondary_anchor_count=config.gating_secondary_anchor_count,
        score_range=config.gating_score_range,
        temperature=config.gating_temperature,
        stored_experts_by_server=tuple(tuple(server) for server in deployment),
    )


def generate_task_graphs(config: ExperimentConfig, seeds: RunSeeds, verbose: bool = True, export_artifacts: bool = True):
    specs = build_dag_specs(config.num_dags, config.common_dag_spec)
    graphs = generate_multiple_dags(
        specs=specs,
        output_dir=config.output_dir,
        seed=seeds.dag_seed,
        image_dpi=config.image_dpi,
        num_experts=config.num_experts,
        num_iot_features=config.num_iot_features,
        iot_features_per_node_range=config.iot_features_per_node_range,
        task_deadline_seconds_range=config.task_deadline_seconds_range,
        reconstruction_sigma=config.reconstruction_sigma,
        reconstruction_error_range=config.reconstruction_error_range,
        gating_spec=build_gating_spec(config, seeds),
        num_calibration_samples=config.num_calibration_samples,
        calibration_loss_range=config.calibration_loss_range,
        shared_subtask_ratio=config.shared_subtask_ratio,
        shared_subtask_graph_count_range=config.shared_subtask_graph_count_range,
        shared_subtask_kind_count=config.shared_subtask_kind_count,
        verbose=verbose,
        export_artifacts=export_artifacts,
    )
    if verbose:
        print(f"Run seeds: DAG={seeds.dag_seed}, network={seeds.network_seed}")
        print(f"\nDone: generated {len(graphs)} DAG(s).")
        print(f"Output directory: {config.output_dir.resolve()}")
    if export_artifacts:
        network_paths = export_network_graph_artifacts(config, seeds)
        if verbose:
            print(f"Network graph code: {network_paths['python'].resolve()}")
            print(f"Network graph data: {network_paths['json'].resolve()}")
            print(f"Network graph image: {network_paths['png'].resolve()}")
    return graphs


def common_scheduler_kwargs(config: ExperimentConfig, seeds: RunSeeds) -> dict[str, Any]:
    return {
        "num_experts": config.num_experts,
        "expert_memory_range": config.expert_memory_range,
        "expert_memory_sizes_mb": config.expert_memory_sizes_mb,
        "expert_inference_times_ms": config.expert_inference_times_ms,
        "expert_hidden_dimensions": config.expert_hidden_dimensions,
        "expert_parameter_counts": config.expert_parameter_counts,
        "inference_unit_cost_per_mb": config.c_inf,
        "reasoning_data_sizes_bytes": config.reasoning_data_sizes_bytes,
        "num_iot_features": config.num_iot_features,
        "num_servers": config.num_edge_servers,
        "num_iot_devices": config.num_iot_devices,
        "features_per_device_range": config.iot_features_per_device_range,
        "server_feature_overlap_ratio": config.iot_server_feature_overlap_ratio,
        "global_random_feature_fraction": config.iot_global_random_feature_fraction,
        "experts_per_server": config.experts_per_server,
        "server_processing_rate": config.server_processing_rate,
        "server_gpu_memory_range": config.server_gpu_memory_range,
        "wired_rate_range": config.wired_rate_range,
        "wired_extra_link_probability": config.wired_extra_link_probability,
        "wired_edge_weight_range": config.wired_edge_weight_range,
        "c_bw": config.c_bw,
        "c_act": config.c_act,
        "c_fwd": config.c_fwd,
        "default_feature_bits": config.default_feature_bits,
        "feature_bits_range": config.feature_bits_range,
        "wavelength": config.wavelength,
        "noise_power": config.noise_power,
        "common_power_ratio": config.common_power_ratio,
        "max_device_power": config.max_device_power,
        "area_size": config.area_size,
        "cell_radius": config.cell_radius,
        "num_antennas": config.num_antennas,
        "random_seed": seeds.network_seed,
        "max_group_size": config.max_group_size,
        "min_rate": config.min_rate,
        "loss_threshold": config.loss_threshold,
        "lambda_reconstruction": config.lambda_reconstruction,
        "calibration_alpha": config.calibration_alpha,
        "reconstruction_sigma": config.reconstruction_sigma,
    }


def topk_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "top_k": config.topk_k,
        "rank_by": config.topk_rank_by,
    }


def grouping_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "clusters_per_server": config.location_aware_grouping_clusters_per_server,
        "kmeans_iterations": config.location_aware_grouping_kmeans_iterations,
    }


def wdmoe_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "wdmoe_initial_threshold": config.wdmoe_initial_threshold,
        "wdmoe_threshold_step": config.wdmoe_threshold_step,
        "wdmoe_max_threshold": config.wdmoe_max_threshold,
        "wdmoe_wlr_target_ratio": config.wdmoe_wlr_target_ratio,
    }


def our_algorithm_kwargs(config: ExperimentConfig, graphs, seeds: RunSeeds) -> dict[str, Any]:
    return {
        "r_sim": random_similarity_relation(graphs, config, seeds),
        "similarity_weight": config.similarity_weight,
        "enable_offline_group_pruning": config.enable_offline_group_pruning,
        "server_spread_penalty_multiplier": config.jrgep_server_spread_penalty_multiplier,
        "predecessor_penalty_multiplier": config.jrgep_predecessor_penalty_multiplier,
    }


def random_similarity_relation(graphs, config: ExperimentConfig, seeds: RunSeeds) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    tasks = graphs_to_tasks(graphs, config.loss_threshold)
    nodes_by_task: dict[str, list[tuple[tuple[str, str], tuple[float, ...]]]] = {}
    for task in tasks:
        task_id = str(task["id"])
        nodes_by_task[task_id] = []
        for subtask in task.get("subtasks", []):
            key = (task_id, str(subtask["id"]))
            gating = tuple(float(value) for value in subtask.get("gating_weights", []))
            if gating:
                nodes_by_task[task_id].append((key, gating))
    task_ids = [task_id for task_id, nodes in nodes_by_task.items() if nodes]
    if len(task_ids) < 2:
        return []

    scored_candidates: list[tuple[float, tuple[tuple[str, str], tuple[str, str]]]] = []
    for left_index, left_task in enumerate(task_ids):
        for right_task in task_ids[left_index + 1 :]:
            for left_node, left_gating in nodes_by_task[left_task]:
                for right_node, right_gating in nodes_by_task[right_task]:
                    similarity = cosine_similarity(left_gating, right_gating)
                    scored_candidates.append((similarity, (left_node, right_node)))
    if not scored_candidates:
        return []

    rng = random.Random((seeds.dag_seed * 1_000_003) ^ seeds.network_seed ^ 0x51A11)
    pair_count = config.random_similarity_pair_count
    if pair_count is None:
        total_nodes = sum(len(nodes) for nodes in nodes_by_task.values())
        pair_count = max(1, total_nodes // 4)
    pair_count = max(0, min(int(pair_count), len(scored_candidates)))
    if pair_count == 0:
        return []

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    pool_size = min(len(scored_candidates), max(pair_count * 5, pair_count))
    top_pool = [pair for _, pair in scored_candidates[:pool_size]]
    return rng.sample(top_pool, pair_count)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def safe_usage(cost: float, unit_cost: float) -> float:
    if unit_cost == 0:
        return 0.0
    return cost / unit_cost


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def print_cost_line(label: str, cost: float, unit_cost: float, usage_unit: str) -> None:
    usage = safe_usage(cost, unit_cost)
    print(
        f"  {label}: usage={format_number(usage)} {usage_unit}, "
        f"unit_cost={unit_cost:g}, cost={rounded_cost(cost)}"
    )


def result_diagnostics(result: Any, config: ExperimentConfig) -> dict[str, float]:
    assignments = getattr(result, "subtask_assignment", {}) or {}
    server_counts: list[int] = []
    expert_counts: list[int] = []
    for pairs in assignments.values():
        pairs = list(pairs)
        server_counts.append(len({server_id for server_id, _ in pairs}))
        expert_counts.append(len(pairs))

    evaluation = getattr(result, "evaluation", None)
    timing = getattr(evaluation, "timing", None)
    start_times = getattr(timing, "subtask_start_time", {}) or {}
    finish_times = getattr(timing, "subtask_finish_time", {}) or {}
    inference_time_ms = 1000.0 * sum(
        max(0.0, finish_time - start_times.get(key, finish_time))
        for key, finish_time in finish_times.items()
    )

    return {
        "activation_usage": safe_usage(result.activation_cost, config.c_act),
        "bandwidth_usage": safe_usage(result.bandwidth_cost, config.c_bw),
        "forwarding_usage": safe_usage(result.forwarding_cost, config.c_fwd),
        "inference_time_ms": inference_time_ms,
        "avg_servers_per_subtask": sum(server_counts) / len(server_counts) if server_counts else 0.0,
        "avg_experts_per_subtask": sum(expert_counts) / len(expert_counts) if expert_counts else 0.0,
    }


def print_diagnostics(result: Any, config: ExperimentConfig) -> None:
    diagnostics = result_diagnostics(result, config)
    print("Diagnostics:")
    print(
        "  Assignment: "
        f"activation_memory={format_number(diagnostics['activation_usage'])}, "
        f"avg_servers/subtask={diagnostics['avg_servers_per_subtask']:.2f}, "
        f"avg_experts/subtask={diagnostics['avg_experts_per_subtask']:.2f}"
    )
    print(
        "  Usage: "
        f"bandwidth={format_number(diagnostics['bandwidth_usage'])} Hz, "
        f"forwarding_events={format_number(diagnostics['forwarding_usage'])}"
    )

def print_method_result(method_name: str, result_path, result: Any, config: ExperimentConfig, graphs=None) -> None:
    print(f"{method_name} result: {result_path.resolve()}")
    print(f"{method_name} total cost: {rounded_cost(result.total_cost)}")
    print("Cost breakdown:")
    print_cost_line("Activation", result.activation_cost, config.c_act, "activated expert memory")
    print_cost_line("Bandwidth", result.bandwidth_cost, config.c_bw, "Hz")
    print_cost_line("Forwarding", result.forwarding_cost, config.c_fwd, "forwarding events")
    print_cost_line(
        "Inference",
        getattr(result, "inference_cost", 0.0),
        config.c_inf,
        "executed expert complexity (MFLOP-equivalent)",
    )
    print_diagnostics(result, config)
    print(f"{method_name} violations: {len(result.violations)}")
    if method_name == METHOD_DISPLAY_NAMES["our_similarity_aware_topk_backhaul"] and graphs is not None:
        print_node_loss_summary(result, config, graphs)


def print_node_loss_summary(result: Any, config: ExperimentConfig, graphs) -> None:
    tasks = normalize_tasks(graphs_to_tasks(graphs, config.loss_threshold))
    evaluator = FormulationEvaluator(
        servers=[],
        devices=[],
        experts={},
        tasks=tasks,
        config=FormulationConfig(
            default_loss_threshold=config.loss_threshold if config.loss_threshold is not None else 3.0,
            lambda_reconstruction=config.lambda_reconstruction,
            calibration_alpha=config.calibration_alpha,
            reconstruction_sigma=config.reconstruction_sigma,
        ),
    )
    print("Node performance-loss summary:")
    print("  node                 prob      rec_loss    perf_loss   threshold   ok")
    all_ok = True
    for task in tasks.values():
        for subtask in task.subtasks:
            label = f"{task.id}:{subtask.id}"
            threshold = evaluator.conformal_loss_threshold(subtask)
            probability = float(result.selected_probability.get(label, 0.0))
            rec_loss = float(result.reconstruction_loss.get(label, 0.0))
            perf_loss = float(result.performance_loss.get(label, float("inf")))
            ok = perf_loss <= threshold + 1e-9
            all_ok = all_ok and ok
            print(
                f"  {label:<20} "
                f"{probability:>8.4f} "
                f"{rec_loss:>10.4f} "
                f"{perf_loss:>10.4f} "
                f"{threshold:>10.4f} "
                f"{'OK' if ok else 'FAIL'}"
            )
    print(f"  All nodes satisfy performance loss: {all_ok}")


METHOD_DISPLAY_NAMES = {
    "hybrid_topk_gssgd_backhaul": "Top-k + CSSGD",
    "hybrid_topk_location_aware_backhaul": "Top-k + Location_aware",
    "wdmoe_gssgd": "WDMoE + CSSGD",
    "wdmoe_location_aware": "WDMoE + Location_aware",
    "our_similarity_aware_topk_backhaul": "JRGXP",
}


COST_COLORS = {
    "Activation": "#4C78A8",
    "Bandwidth": "#F58518",
    "Forwarding": "#54A24B",
    "Inference": "#B279A2",
}


def sanitize_filename(value: str) -> str:
    cleaned = []
    for char in value.lower().strip():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_", "+", "(" , ")"}:
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "figure"


def save_figure(fig: Any, figure_dir: Path, filename: str) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return output_path


def show_total_cost_plot(
    results: dict[str, Any],
    title: str = "Average Total Cost Comparison",
    figure_dir: Path | None = None,
    filename: str | None = None,
) -> Path | None:
    if not results:
        return None

    import matplotlib.pyplot as plt

    method_names = [METHOD_DISPLAY_NAMES.get(key, key) for key in results]
    total_costs = [rounded_cost(result.total_cost) for result in results.values()]

    fig, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(method_names, total_costs, color="#4C78A8")

    axis.set_title(title)
    axis.set_ylabel("Cost")
    axis.tick_params(axis="x", rotation=18)
    axis.bar_label(bars, padding=3)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    saved_path = None
    if figure_dir is not None:
        saved_path = save_figure(
            fig,
            figure_dir,
            filename or f"{sanitize_filename(title)}.png",
        )
        print(f"Saved figure: {saved_path.resolve()}")
    plt.show()
    return saved_path


def show_cost_breakdown_plot(
    results: dict[str, Any],
    title: str = "Cost Breakdown",
    figure_dir: Path | None = None,
    filename: str | None = None,
) -> Path | None:
    if not results:
        return None

    import matplotlib.pyplot as plt

    method_names = [METHOD_DISPLAY_NAMES.get(key, key) for key in results]
    fig, axis = plt.subplots(figsize=(9, 5.6))
    bottoms = [0 for _ in method_names]
    for label in ("Activation", "Bandwidth", "Forwarding", "Inference"):
        values = [rounded_cost(getattr(result, f"{label.lower()}_cost")) for result in results.values()]
        bars = axis.bar(
            method_names,
            values,
            bottom=bottoms,
            label=label,
            color=COST_COLORS[label],
        )
        labels = [str(value) if value else "" for value in values]
        axis.bar_label(bars, labels=labels, label_type="center")
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]


    for index, (bottom, result) in enumerate(zip(bottoms, results.values())):
        violation_count = len(getattr(result, "violations", []))
        if violation_count:
            axis.text(
                index,
                bottom,
                f"violations={violation_count}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="crimson",
                fontweight="bold",
            )
    axis.set_title(title)
    axis.set_ylabel("Cost")
    axis.tick_params(axis="x", rotation=18)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    saved_path = None
    if figure_dir is not None:
        saved_path = save_figure(
            fig,
            figure_dir,
            filename or f"{sanitize_filename(title)}.png",
        )
        print(f"Saved figure: {saved_path.resolve()}")
    plt.show()
    return saved_path


def show_cost_plot(
    results: dict[str, Any],
    title: str = "Cost Comparison",
    figure_dir: Path | None = None,
    filename: str | None = None,
) -> Path | None:
    if filename:
        return show_total_cost_plot(results, title, figure_dir, filename)
    show_total_cost_plot(results, title, figure_dir, "average_total_cost_comparison.png")
    return show_cost_breakdown_plot(results, "Cost Breakdown", figure_dir, "cost_breakdown_stacked_bar.png")


def run_enabled_methods(
    config: ExperimentConfig,
    graphs,
    seeds: RunSeeds,
    *,
    print_details: bool = True,
    show_plot: bool = True,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    base_kwargs = common_scheduler_kwargs(config, seeds)
    rsma_integration.NESTED_SERVER_BASE = config.nested_server_base
    expert_deployment.NESTED_STORAGE_BASE = config.nested_storage_base

    if config.run_hybrid_topk_gssgd_backhaul:
        result_path = config.results_dir / "hybrid_topk_gssgd_backhaul_result.json"
        result = run_hybrid_topk_gssgd_backhaul(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
        )
        if print_details:
            print_method_result(METHOD_DISPLAY_NAMES["hybrid_topk_gssgd_backhaul"], result_path, result, config, graphs)
        results["hybrid_topk_gssgd_backhaul"] = result

    if config.run_hybrid_topk_location_aware_backhaul:
        result_path = config.results_dir / "hybrid_topk_location_aware_result.json"
        result = run_hybrid_topk_location_aware_backhaul(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
        )
        if print_details:
            print_method_result(METHOD_DISPLAY_NAMES["hybrid_topk_location_aware_backhaul"], result_path, result, config, graphs)
        results["hybrid_topk_location_aware_backhaul"] = result


    if config.run_wdmoe_gssgd:
        result_path = config.results_dir / "wdmoe_gssgd_result.json"
        result = run_wdmoe_gssgd(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
            **wdmoe_kwargs(config),
        )
        if print_details:
            print_method_result(METHOD_DISPLAY_NAMES["wdmoe_gssgd"], result_path, result, config, graphs)
        results["wdmoe_gssgd"] = result

    if config.run_wdmoe_location_aware:
        result_path = config.results_dir / "wdmoe_location_aware_result.json"
        result = run_wdmoe_location_aware(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
            **wdmoe_kwargs(config),
        )
        if print_details:
            print_method_result(METHOD_DISPLAY_NAMES["wdmoe_location_aware"], result_path, result, config, graphs)
        results["wdmoe_location_aware"] = result

    if config.run_our_similarity_aware_topk_backhaul:
        result_path = config.results_dir / "jrgep_result.json"
        result = run_similarity_aware_topk_backhaul(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **our_algorithm_kwargs(config, graphs, seeds),
        )
        if print_details:
            print_method_result(METHOD_DISPLAY_NAMES["our_similarity_aware_topk_backhaul"], result_path, result, config, graphs)
        results["our_similarity_aware_topk_backhaul"] = result

    if show_plot:
        figure_dir = config.results_dir / "figure"
        show_total_cost_plot(
            results,
            title="Current Run Total Cost Comparison",
            figure_dir=figure_dir,
            filename="current_run_total_cost_comparison.png",
        )
        show_cost_breakdown_plot(
            results,
            title="Current Run Cost Breakdown",
            figure_dir=figure_dir,
            filename="current_run_cost_breakdown_stacked_bar.png",
        )
    return results


def print_run_summary(run_number: int, results: dict[str, Any], config: ExperimentConfig) -> None:
    parts = []
    for method_key, result in results.items():
        name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        diagnostics = result_diagnostics(result, config)
        parts.append(
            f"{name}: total={rounded_cost(result.total_cost)}, "
            f"act_mem={format_number(diagnostics['activation_usage'])}, "
            f"bw={format_number(diagnostics['bandwidth_usage'])}Hz, "
            f"fwd={format_number(diagnostics['forwarding_usage'])}, "
            f"srv/sub={diagnostics['avg_servers_per_subtask']:.2f}, "
            f"violations={len(result.violations)}"
        )
    print(f"Run {run_number}: " + " | ".join(parts))

def average_run_results(run_results: list[dict[str, Any]], config: ExperimentConfig) -> dict[str, AverageResult]:
    if not run_results:
        return {}

    method_keys = list(run_results[0].keys())
    averages: dict[str, AverageResult] = {}
    for method_key in method_keys:
        method_results = [result_set[method_key] for result_set in run_results if method_key in result_set]
        count = len(method_results)
        if count == 0:
            continue
        diagnostics = [result_diagnostics(result, config) for result in method_results]
        averages[method_key] = AverageResult(
            total_cost=sum(result.total_cost for result in method_results) / count,
            activation_cost=sum(result.activation_cost for result in method_results) / count,
            bandwidth_cost=sum(result.bandwidth_cost for result in method_results) / count,
            forwarding_cost=sum(result.forwarding_cost for result in method_results) / count,
            inference_cost=sum(getattr(result, "inference_cost", 0.0) for result in method_results) / count,
            violation_count=sum(len(result.violations) for result in method_results) / count,
            activation_usage=sum(item["activation_usage"] for item in diagnostics) / count,
            bandwidth_usage=sum(item["bandwidth_usage"] for item in diagnostics) / count,
            forwarding_usage=sum(item["forwarding_usage"] for item in diagnostics) / count,
            inference_time_ms=sum(item["inference_time_ms"] for item in diagnostics) / count,
            avg_servers_per_subtask=sum(item["avg_servers_per_subtask"] for item in diagnostics) / count,
            avg_experts_per_subtask=sum(item["avg_experts_per_subtask"] for item in diagnostics) / count,
        )
    return averages


def print_average_results(averages: dict[str, AverageResult], config: ExperimentConfig, num_runs: int) -> None:
    print(f"\nAverage result over {num_runs} run(s)")
    print("=" * 72)
    for method_key, result in averages.items():
        method_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        print(f"{method_name} average total cost: {result.total_cost:.2f} ({rounded_cost(result.total_cost)})")
        print("Average cost breakdown:")
        print_cost_line("Activation", result.activation_cost, config.c_act, "activated expert memory")
        print_cost_line("Bandwidth", result.bandwidth_cost, config.c_bw, "Hz")
        print_cost_line("Forwarding", result.forwarding_cost, config.c_fwd, "forwarding events")
        print_cost_line(
            "Inference",
            result.inference_cost,
            config.c_inf,
            "executed expert complexity (MFLOP-equivalent)",
        )
        print("Average diagnostics:")
        print(
            "  Assignment: "
            f"activation_memory={result.activation_usage:.2f}, "
            f"avg_servers/subtask={result.avg_servers_per_subtask:.2f}, "
            f"avg_experts/subtask={result.avg_experts_per_subtask:.2f}"
        )
        print(
            "  Usage: "
            f"bandwidth={result.bandwidth_usage:.2f} Hz, "
            f"forwarding_events={result.forwarding_usage:.2f}"
        )
        print(f"{method_name} average violations: {result.violation_count:.2f}")
        print("-" * 72)


def run_single_experiment(
    config: ExperimentConfig,
    *,
    run_index: int = 0,
    verbose: bool = True,
    print_details: bool = True,
    show_plot: bool = True,
) -> dict[str, Any]:
    seeds = resolve_run_seeds(config, run_index=run_index)
    graphs = generate_task_graphs(config, seeds, verbose=verbose, export_artifacts=verbose)
    return run_enabled_methods(
        config,
        graphs,
        seeds,
        print_details=print_details,
        show_plot=show_plot,
    )


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    if config.num_runs <= 1:
        return run_single_experiment(config)

    print(f"Running {config.num_runs} independent run(s) for average results.")
    print(f"Each run generates {config.num_dags} DAG(s).")
    run_results: list[dict[str, Any]] = []
    for run_index in range(config.num_runs):
        seeds = resolve_run_seeds(config, run_index=run_index)
        print(
            f"\nRun {run_index + 1}/{config.num_runs}: "
            f"DAG seed={seeds.dag_seed}, network seed={seeds.network_seed}"
        )
        graphs = generate_task_graphs(config, seeds, verbose=False, export_artifacts=(run_index == 0))
        results = run_enabled_methods(
            config,
            graphs,
            seeds,
            print_details=False,
            show_plot=False,
        )
        print_run_summary(run_index + 1, results, config)
        run_results.append(results)

    averages = average_run_results(run_results, config)
    print_average_results(averages, config, config.num_runs)
    show_cost_plot(
        averages,
        title=f"Average Cost Comparison ({config.num_runs} runs)",
        figure_dir=config.results_dir / "figure",
        filename=None,
    )
    return averages

