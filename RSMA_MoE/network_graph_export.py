from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import ExperimentConfig
from rsma_integration import build_feature_bits, build_simple_experts, build_simple_servers


def build_current_network_graph(config: ExperimentConfig, seeds: Any) -> tuple[list[dict[str, Any]], nx.Graph]:
    """Rebuild the physical wired graph with the same RNG order used by schedulers."""
    rng = random.Random(seeds.network_seed)

    build_feature_bits(
        config.num_iot_features,
        config.default_feature_bits,
        config.feature_bits_range,
        rng,
    )
    experts = build_simple_experts(
        config.num_experts,
        memory_range=config.expert_memory_range,
        memory_sizes_mb=config.expert_memory_sizes_mb,
        inference_times_ms=config.expert_inference_times_ms,
        hidden_dimensions=config.expert_hidden_dimensions,
        parameter_counts=config.expert_parameter_counts,
        inference_unit_cost_per_mb=config.c_inf,
        rng=rng,
    )
    servers = build_simple_servers(
        num_servers=config.num_edge_servers,
        experts=experts,
        experts_per_server=config.experts_per_server,
        processing_rate=config.server_processing_rate,
        gpu_memory_range=config.server_gpu_memory_range,
        wired_rate_range=config.wired_rate_range,
        wired_extra_link_probability=config.wired_extra_link_probability,
        wired_edge_weight_range=config.wired_edge_weight_range,
        rng=rng,
        deployment_seed=seeds.network_seed,
    )

    graph = nx.Graph()
    for server in servers:
        graph.add_node(server["id"])
    for link in servers[0].get("physical_wired_links", []):
        graph.add_edge(
            link["src"],
            link["dst"],
            rate=float(link["rate"]),
            weight=float(link["weight"]),
        )
    return servers, graph


def export_network_graph_artifacts(config: ExperimentConfig, seeds: Any) -> dict[str, Path]:
    """Write network graph JSON, GraphML, Python code, and PNG into generated_dags."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    servers, graph = build_current_network_graph(config, seeds)
    server_ids = [f"server_{index}" for index in range(config.num_edge_servers)]
    server_by_id = {server["id"]: server for server in servers}

    route_weights: dict[str, dict[str, float]] = {}
    route_rates: dict[str, dict[str, float]] = {}
    for src in server_ids:
        route_weights[src] = {
            dst: float(server_by_id[src]["wired_weights"][dst])
            for dst in server_ids
            if dst != src
        }
        route_rates[src] = {
            dst: float(server_by_id[src]["wired_rates"][dst])
            for dst in server_ids
            if dst != src
        }

    physical_links = [
        {
            "src": src,
            "dst": dst,
            "rate_bps": float(data["rate"]),
            "weight": float(data["weight"]),
        }
        for src, dst, data in graph.edges(data=True)
    ]
    payload = {
        "network_seed": seeds.network_seed,
        "num_servers": config.num_edge_servers,
        "wired_rate_range_bps": list(config.wired_rate_range),
        "wired_extra_link_probability": config.wired_extra_link_probability,
        "wired_edge_weight_range": list(config.wired_edge_weight_range),
        "physical_links": physical_links,
        "shortest_route_cost_weights": route_weights,
        "shortest_route_bottleneck_rates_bps": route_rates,
    }

    json_path = output_dir / "network_graph.json"
    graphml_path = output_dir / "network_graph.graphml"
    code_path = output_dir / "network_graph.py"
    png_path = output_dir / "network_graph.png"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    nx.write_graphml(graph, graphml_path)
    code_path.write_text(_network_graph_code(payload), encoding="utf-8")
    _draw_network_graph(graph, server_ids, route_weights, route_rates, config, png_path)

    return {
        "json": json_path,
        "graphml": graphml_path,
        "python": code_path,
        "png": png_path,
    }


def _network_graph_code(payload: dict[str, Any]) -> str:
    return (
        "from __future__ import annotations\n\n"
        "# Auto-generated MEC network graph artifact.\n"
        "# Run main.py again to regenerate this file with the current config/seed.\n\n"
        f"NETWORK_SEED = {payload['network_seed']!r}\n"
        f"NUM_SERVERS = {payload['num_servers']!r}\n"
        f"WIRED_RATE_RANGE_BPS = {payload['wired_rate_range_bps']!r}\n"
        f"WIRED_EXTRA_LINK_PROBABILITY = {payload['wired_extra_link_probability']!r}\n"
        f"WIRED_EDGE_WEIGHT_RANGE = {payload['wired_edge_weight_range']!r}\n"
        f"PHYSICAL_LINKS = {payload['physical_links']!r}\n"
        f"SHORTEST_ROUTE_COST_WEIGHTS = {payload['shortest_route_cost_weights']!r}\n"
        f"SHORTEST_ROUTE_BOTTLENECK_RATES_BPS = {payload['shortest_route_bottleneck_rates_bps']!r}\n\n"
        "def physical_edges():\n"
        "    return [(link['src'], link['dst'], link['rate_bps'], link['weight']) for link in PHYSICAL_LINKS]\n\n"
        "def wired_weight(src: str, dst: str) -> float:\n"
        "    if src == dst:\n"
        "        return 0.0\n"
        "    return SHORTEST_ROUTE_COST_WEIGHTS[src][dst]\n\n"
        "def wired_rate(src: str, dst: str) -> float:\n"
        "    if src == dst:\n"
        "        return float('inf')\n"
        "    return SHORTEST_ROUTE_BOTTLENECK_RATES_BPS[src][dst]\n"
    )


def _draw_network_graph(
    graph: nx.Graph,
    server_ids: list[str],
    route_weights: dict[str, dict[str, float]],
    route_rates: dict[str, dict[str, float]],
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    weights = np.zeros((len(server_ids), len(server_ids)))
    rates = np.zeros((len(server_ids), len(server_ids)))
    for row, src in enumerate(server_ids):
        for col, dst in enumerate(server_ids):
            if src == dst:
                weights[row, col] = 0.0
                rates[row, col] = np.inf
            else:
                weights[row, col] = route_weights[src][dst]
                rates[row, col] = route_rates[src][dst]

    fig = plt.figure(figsize=(15, 7), dpi=config.image_dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_graph = fig.add_subplot(grid[0, 0])
    ax_matrix = fig.add_subplot(grid[0, 1])

    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1600,
        node_color="#4C78A8",
        edgecolors="#1f2d3a",
        linewidths=1.2,
        ax=ax_graph,
    )
    nx.draw_networkx_edges(graph, pos, width=2.0, edge_color="#666666", ax=ax_graph)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="white", font_weight="bold", ax=ax_graph)
    edge_labels = {
        (src, dst): f"{data['rate'] / 1e9:.2f}G\nw={data['weight']:.2f}"
        for src, dst, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax_graph)
    ax_graph.set_title("Physical MEC Wired Graph\n(edge label = link rate and forwarding weight)")
    ax_graph.axis("off")

    image = ax_matrix.imshow(weights, cmap="Blues", vmin=0)
    ax_matrix.set_title("Logical Forwarding Weight\n(minimum accumulated edge weight)")
    ax_matrix.set_xticks(range(len(server_ids)))
    ax_matrix.set_yticks(range(len(server_ids)))
    ax_matrix.set_xticklabels([server_id.replace("server_", "s") for server_id in server_ids], rotation=45, ha="right")
    ax_matrix.set_yticklabels([server_id.replace("server_", "s") for server_id in server_ids])
    threshold = max(1.5, weights.max() * 0.55)
    for row in range(weights.shape[0]):
        for col in range(weights.shape[1]):
            color = "white" if weights[row, col] >= threshold else "black"
            ax_matrix.text(col, row, f"{weights[row, col]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04, label="route weight")

    finite_rates = rates[np.isfinite(rates)]
    summary = (
        f"servers={config.num_edge_servers} | physical links={graph.number_of_edges()} | "
        f"avg minimum route weight={weights[weights > 0].mean():.2f} | "
        f"max minimum route weight={weights.max():.2f} | "
        f"avg route bottleneck rate={finite_rates.mean() / 1e9:.2f} Gbps"
    )
    fig.suptitle("Current Network Graph Used by the Simulator", fontsize=15, y=0.98)
    fig.text(0.5, 0.02, summary, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
