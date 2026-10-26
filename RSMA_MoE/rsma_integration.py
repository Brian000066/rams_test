from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from physical_layer import TX_PSD_W_PER_HZ
from expert_deployment import build_expert_deployment, normalize_experts_per_server

import networkx as nx



def feature_name(feature_index: int | str) -> str:
    return f"feature_{feature_index}"


def expert_id(expert_index: int | str) -> str:
    return f"expert_{expert_index}"


def feature_index(feature: str) -> int:
    return int(str(feature).split("_")[-1])


def sort_feature_names(features: Iterable[str]) -> list[str]:
    return sorted(set(features), key=feature_index)


def ensure_iot_feature_coverage(
    devices: list[dict[str, Any]],
    feature_pool: Sequence[int],
    server_feature_pools: Mapping[int, Sequence[int]],
    max_features_per_device: int,
    rng: random.Random,
) -> None:
    if not devices:
        return
    covered = {feature_index(feature) for device in devices for feature in device.get("features", [])}
    missing = [feature for feature in feature_pool if feature not in covered]
    if not missing:
        return

    for missing_feature in missing:
        preferred_servers = {
            server_index
            for server_index, pool in server_feature_pools.items()
            if missing_feature in set(pool)
        }
        candidates = [
            device
            for device in devices
            if int(str(device["home_server"]).split("_")[-1]) in preferred_servers
        ] or devices
        under_capacity = [
            device
            for device in candidates
            if len(device.get("features", [])) < max_features_per_device
        ]
        candidates = under_capacity or candidates
        rng.shuffle(candidates)
        target = min(candidates, key=lambda device: len(device.get("features", [])))
        current = set(target.get("features", []))
        current.add(feature_name(missing_feature))
        target["features"] = sort_feature_names(current)

def build_feature_bits(
    num_iot_features: int,
    default_feature_bits: float,
    feature_bits_range: tuple[float, float] | None = None,
    rng: random.Random | None = None,
) -> dict[str, float]:
    if num_iot_features < 1:
        return {}
    rng = rng or random.Random()
    if feature_bits_range is None:
        low = high = float(default_feature_bits)
    else:
        low, high = feature_bits_range
        low = float(low)
        high = float(high)
        if low <= 0.0 or high <= 0.0:
            raise ValueError("feature_bits_range values must be positive.")
        if low > high:
            low, high = high, low
    return {
        feature_name(index): (low if low == high else rng.uniform(low, high))
        for index in range(num_iot_features)
    }


def node_to_subtask(
    node_id: str,
    node: Mapping[str, Any],
    predecessors: Sequence[str],
    loss_threshold: float | None,
    reasoning_data_sizes_bytes: Sequence[int] = (512,),
    default_feature_bits: float = 12000.0,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    required_data = node.get("required_data", {})
    iot_features = required_data.get("iot_features", [])
    gating_weights = list(node.get("gating_weights", []))
    expert_confidence = list(node.get("expert_confidence", []))
    reconstruction_errors = list(node.get("reconstruction_errors", []))
    reconstruction_errors_by_feature = {
        feature_name(feature): float(error)
        for feature, error in zip(iot_features, reconstruction_errors)
    }
    reasoning_sizes = tuple(int(size) for size in reasoning_data_sizes_bytes)
    if not reasoning_sizes or any(size <= 0 for size in reasoning_sizes):
        raise ValueError("reasoning_data_sizes_bytes must contain positive values.")
    rng = rng or random.Random()
    reasoning_data_size_bytes = int(
        node.get("reasoning_data_size_bytes", rng.choice(reasoning_sizes))
    )
    # The setup uses BF16 outputs (16 bits = 2 bytes per token).
    output_tokens = math.ceil(reasoning_data_size_bytes / 2.0)

    return {
        "id": str(node_id),
        "unique_id": str(node.get("unique_subtask_id", node_id)),
        "required_features": [feature_name(index) for index in iot_features],
        "predecessors": [str(item) for item in predecessors],
        "output_tokens": output_tokens,
        "reasoning_data_size_bytes": reasoning_data_size_bytes,
        "feature_volume_bits": float(default_feature_bits) * max(len(iot_features), 1),
        "loss_threshold": loss_threshold,
        "prompt": node.get("prompt", ""),
        "gating_weights": gating_weights,
        "expert_confidence": expert_confidence,
        "reconstruction_loss": float(node.get("reconstruction_loss", 0.0)),
        "reconstruction_errors_by_feature": reconstruction_errors_by_feature,
        "calibration_losses": list(node.get("calibration_losses", [])),
    }


def graphs_to_tasks(
    graphs: Iterable[nx.DiGraph],
    loss_threshold: float | None,
    reasoning_data_sizes_bytes: Sequence[int] = (512,),
    default_feature_bits: float = 12000.0,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Convert generated DAG objects into JRGEP scheduler task specs."""
    tasks: list[dict[str, Any]] = []

    for graph in graphs:
        graph_meta = dict(graph.graph)
        task_id = str(graph_meta.get("task_id", f"task_{graph_meta.get('graph_index', len(tasks) + 1)}"))
        subtasks: list[dict[str, Any]] = []

        for node_id in nx.topological_sort(graph):
            node = graph.nodes[node_id]
            required_data = node.get("required_data", {})
            upstream_outputs = required_data.get("upstream_outputs", [])
            predecessors = [
                str(item.get("source_node"))
                for item in upstream_outputs
                if item.get("source_node") is not None
            ]
            if not predecessors:
                predecessors = [str(pred) for pred in graph.predecessors(node_id)]
            subtasks.append(
                node_to_subtask(
                    str(node_id),
                    node,
                    predecessors,
                    loss_threshold,
                    reasoning_data_sizes_bytes=reasoning_data_sizes_bytes,
                    default_feature_bits=default_feature_bits,
                    rng=rng,
                )
            )

        tasks.append(
            {
                "id": task_id,
                "deadline": float(graph_meta.get("deadline_seconds", float("inf"))),
                "subtasks": subtasks,
            }
        )

    return tasks


def load_generated_dag_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dag_json_to_task(
    data: Mapping[str, Any],
    loss_threshold: float | None = None,
) -> dict[str, Any]:
    """Convert one saved generated_dags/dag_N.json file to a scheduler task."""
    graph_meta = dict(data.get("graph", {}))
    task_id = str(graph_meta.get("task_id", graph_meta.get("graph_index", "task")))

    predecessors_by_node: dict[str, list[str]] = {}
    for edge in data.get("edges", []):
        predecessors_by_node.setdefault(str(edge["target"]), []).append(str(edge["source"]))

    subtasks: list[dict[str, Any]] = []
    for node in data.get("nodes", []):
        node_id = str(node.get("id"))
        required_data = node.get("required_data", {})
        upstream_outputs = required_data.get("upstream_outputs", [])
        predecessors = [
            str(item.get("source_node"))
            for item in upstream_outputs
            if item.get("source_node") is not None
        ] or predecessors_by_node.get(node_id, [])
        subtasks.append(
            node_to_subtask(
                node_id,
                node,
                predecessors,
                loss_threshold,
            )
        )

    return {
        "id": task_id,
        "deadline": float(graph_meta.get("deadline_seconds", float("inf"))),
        "subtasks": subtasks,
    }


def load_tasks_from_generated_dir(
    generated_dir: Path,
    loss_threshold: float | None = None,
) -> list[dict[str, Any]]:
    return [
        dag_json_to_task(load_generated_dag_json(path), loss_threshold)
        for path in sorted(generated_dir.glob("dag_*.json"))
    ]


def normalize_float_range(
    value_range: tuple[float, float] | None,
    default: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    low, high = value_range or default
    low = float(low)
    high = float(high)
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"{label} values must be positive.")
    if low > high:
        low, high = high, low
    return low, high


def build_simple_experts(
    num_experts: int,
    memory_range: tuple[float, float] | None = None,
    memory_sizes_mb: Sequence[float] | None = None,
    inference_times_ms: Sequence[float] | None = None,
    hidden_dimensions: Sequence[int] | None = None,
    parameter_counts: Sequence[int] | None = None,
    inference_unit_cost_per_mb: float = 0.06,
    rng: random.Random | None = None,
) -> dict[str, dict[str, float | int]]:
    """Create indexed experts with heterogeneous model sizes."""
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1.")

    rng = rng or random.Random()
    memory_values: tuple[float, ...] | None = None
    if memory_sizes_mb is not None:
        memory_values = tuple(float(value) for value in memory_sizes_mb)
        if len(memory_values) != num_experts:
            raise ValueError("expert_memory_sizes_mb length must match num_experts.")
        if any(value <= 0.0 for value in memory_values):
            raise ValueError("expert_memory_sizes_mb values must be positive.")
        memory_min = memory_max = 0.0
    else:
        memory_min, memory_max = normalize_float_range(
            memory_range,
            (256.0, 256.0),
            "expert_memory_range",
        )
    if inference_times_ms is not None and len(inference_times_ms) != num_experts:
        raise ValueError("expert_inference_times_ms length must match num_experts.")
    if inference_times_ms is not None and any(value <= 0.0 for value in inference_times_ms):
        raise ValueError("expert_inference_times_ms values must be positive.")
    if hidden_dimensions is not None and len(hidden_dimensions) != num_experts:
        raise ValueError("expert_hidden_dimensions length must match num_experts.")
    if hidden_dimensions is not None and any(value <= 0 for value in hidden_dimensions):
        raise ValueError("expert_hidden_dimensions values must be positive.")
    if parameter_counts is not None and len(parameter_counts) != num_experts:
        raise ValueError("expert_parameter_counts length must match num_experts.")
    if parameter_counts is not None and any(value <= 0 for value in parameter_counts):
        raise ValueError("expert_parameter_counts values must be positive.")
    if inference_unit_cost_per_mb < 0.0:
        raise ValueError("inference unit cost cannot be negative.")

    experts: dict[str, dict[str, float | int]] = {}
    for index in range(num_experts):
        memory = (
            memory_values[index]
            if memory_values is not None
            else (
                memory_min
                if memory_min == memory_max
                else rng.uniform(memory_min, memory_max)
            )
        )
        # Keep the public argument name for backward compatibility.  Under the
        # ICC27 formulation it is interpreted as the unit cost per one million
        # computational-complexity units, rather than a cost per MB.
        complexity_mflop_equivalent = (
            float(parameter_counts[index]) / 1_000_000.0
            if parameter_counts is not None
            else memory
        )
        experts[expert_id(index)] = {
            "index": index,
            "hidden_dimension": int(hidden_dimensions[index]) if hidden_dimensions is not None else 0,
            "parameter_count": int(parameter_counts[index]) if parameter_counts is not None else 0,
            "memory": memory,
            "latency": (
                float(inference_times_ms[index]) / 1000.0
                if inference_times_ms is not None
                else 0.05 + index * 0.01
            ),
            "inference_cost": (
                complexity_mflop_equivalent
                * float(inference_unit_cost_per_mb)
            ),
        }
    return experts



def build_physical_wired_shortest_paths(
    num_servers: int,
    rate_min: float,
    rate_max: float,
    rng: random.Random,
    extra_link_probability: float = 0.25,
    edge_weight_range: tuple[float, float] = (1.0, 3.0),
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]], list[dict[str, Any]]]:
    """Build a connected physical MEC graph and derive minimum-cost routes.

    The logical wired connection between any two servers follows the
    physical route with the smallest sum of edge weights. The effective
    end-to-end rate is the bottleneck rate along that selected route.
    """
    if num_servers < 1:
        return {}, {}, []

    weight_min, weight_max = edge_weight_range
    if weight_min <= 0.0 or weight_max <= 0.0:
        raise ValueError("wired edge weights must be positive.")
    if weight_min > weight_max:
        weight_min, weight_max = weight_max, weight_min

    links: dict[tuple[int, int], tuple[float, float]] = {}

    def add_link(left: int, right: int) -> None:
        if left == right:
            return
        key = (min(left, right), max(left, right))
        if key not in links:
            links[key] = (
                rng.uniform(rate_min, rate_max),
                rng.uniform(weight_min, weight_max),
            )

    if num_servers == 1:
        return {0: {}}, {0: {}}, []

    # Ring links guarantee connectivity; random chords create heterogeneous routes.
    for index in range(num_servers):
        add_link(index, (index + 1) % num_servers)
    for left in range(num_servers):
        for right in range(left + 2, num_servers):
            if num_servers > 3 and left == 0 and right == num_servers - 1:
                continue
            if rng.random() < extra_link_probability:
                add_link(left, right)

    route_cost = [[math.inf for _ in range(num_servers)] for _ in range(num_servers)]
    hops = [[math.inf for _ in range(num_servers)] for _ in range(num_servers)]
    bottleneck = [[0.0 for _ in range(num_servers)] for _ in range(num_servers)]
    for index in range(num_servers):
        route_cost[index][index] = 0.0
        hops[index][index] = 0
        bottleneck[index][index] = math.inf
    for (left, right), (rate, edge_weight) in links.items():
        route_cost[left][right] = route_cost[right][left] = edge_weight
        hops[left][right] = hops[right][left] = 1
        bottleneck[left][right] = bottleneck[right][left] = rate

    for mid in range(num_servers):
        for src in range(num_servers):
            if route_cost[src][mid] == math.inf:
                continue
            for dst in range(num_servers):
                if route_cost[mid][dst] == math.inf:
                    continue
                candidate_cost = route_cost[src][mid] + route_cost[mid][dst]
                candidate_hops = hops[src][mid] + hops[mid][dst]
                candidate_rate = min(bottleneck[src][mid], bottleneck[mid][dst])
                cost_tied = math.isclose(
                    candidate_cost,
                    route_cost[src][dst],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                if (
                    candidate_cost < route_cost[src][dst] - 1e-12
                    or (
                        cost_tied
                        and (
                            candidate_rate > bottleneck[src][dst]
                            or (
                                math.isclose(candidate_rate, bottleneck[src][dst])
                                and candidate_hops < hops[src][dst]
                            )
                        )
                    )
                ):
                    route_cost[src][dst] = candidate_cost
                    hops[src][dst] = candidate_hops
                    bottleneck[src][dst] = candidate_rate

    wired_rates: dict[int, dict[int, float]] = {index: {} for index in range(num_servers)}
    wired_weights: dict[int, dict[int, float]] = {index: {} for index in range(num_servers)}
    for src in range(num_servers):
        for dst in range(num_servers):
            if src == dst:
                continue
            wired_rates[src][dst] = bottleneck[src][dst] if bottleneck[src][dst] > 0.0 else rate_min
            wired_weights[src][dst] = float(
                route_cost[src][dst]
                if route_cost[src][dst] != math.inf
                else num_servers * weight_max
            )

    link_list = [
        {
            "src": f"server_{left}",
            "dst": f"server_{right}",
            "rate": rate,
            "weight": edge_weight,
        }
        for (left, right), (rate, edge_weight) in sorted(links.items())
    ]
    return wired_rates, wired_weights, link_list

_PROCESSING_RATE_SALT = 0x9C0DE5


def server_processing_rate_for(
    server_index: int,
    processing_rate: float | Sequence[float],
    seed: int,
) -> float:
    """Processing rate mu_s of one server (parameters per second).

    A single number gives every server the same rate.  A (min, max) range
    gives server k its own rate ~ U[min, max], drawn from a dedicated RNG
    keyed by (seed, k): it does not consume the shared network RNG, and
    server k gets the same rate for every server count (nested Fig. 3).
    """
    if isinstance(processing_rate, (int, float)):
        rate = float(processing_rate)
    else:
        low, high = (float(value) for value in processing_rate)
        if low > high:
            low, high = high, low
        rate = low if low == high else random.Random(
            (int(seed) * 1_000_003 + int(server_index) * 104_729) ^ _PROCESSING_RATE_SALT
        ).uniform(low, high)
    if rate <= 0.0:
        raise ValueError("server processing_rate must be positive.")
    return rate


def build_simple_servers(
    num_servers: int,
    experts: Mapping[str, Mapping[str, Any]],
    experts_per_server: int | Sequence[int],
    processing_rate: float | Sequence[float] = 300e6,
    gpu_memory: float = 8192.0,
    gpu_memory_range: tuple[float, float] | None = None,
    wired_rate_range: tuple[float, float] | None = None,
    wired_extra_link_probability: float = 0.05,
    wired_edge_weight_range: tuple[float, float] = (1.0, 3.0),
    rng: random.Random | None = None,
    deployment_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Create edge servers with the shared expert deployment and random wired links.

    The stored-expert sets come from expert_deployment.build_expert_deployment
    with ``deployment_seed`` so that DAG generation (gating anchors), every
    scheduler and the network export all see the same deployment.
    """
    if num_servers < 1:
        raise ValueError("num_servers must be at least 1.")
    rng = rng or random.Random()
    expert_ids = sorted(experts, key=lambda item: int(item.rsplit("_", 1)[1]))
    if not expert_ids:
        raise ValueError("at least one expert is required.")
    if deployment_seed is None:
        deployment_seed = rng.randint(1, 2_147_483_647)

    deployment = build_expert_deployment(
        num_experts=len(expert_ids),
        num_servers=num_servers,
        experts_per_server=experts_per_server,
        seed=deployment_seed,
    )
    experts_by_server: list[list[str]] = [
        [expert_ids[index] for index in server_experts]
        for server_experts in deployment
    ]

    gpu_min, gpu_max = normalize_float_range(gpu_memory_range, (gpu_memory, gpu_memory), "server_gpu_memory_range")
    rate_min, rate_max = wired_rate_range or (1e9, 1e9)
    if rate_min <= 0.0 or rate_max <= 0.0:
        raise ValueError("wired rates must be positive.")
    if rate_min > rate_max:
        rate_min, rate_max = rate_max, rate_min

    wired_rates, wired_weights, physical_links = build_physical_wired_shortest_paths(
        num_servers,
        rate_min,
        rate_max,
        rng,
        extra_link_probability=wired_extra_link_probability,
        edge_weight_range=wired_edge_weight_range,
    )

    servers: list[dict[str, Any]] = []
    for server_index, stored in enumerate(experts_by_server):
        server_gpu_memory = gpu_min if gpu_min == gpu_max else rng.uniform(gpu_min, gpu_max)
        servers.append(
            {
                "id": f"server_{server_index}",
                "processing_rate": server_processing_rate_for(server_index, processing_rate, deployment_seed),
                "gpu_memory": server_gpu_memory,
                "stored_experts": stored,
                "active_experts": [],
                "wired_rates": {
                    f"server_{other}": wired_rates[server_index][other]
                    for other in range(num_servers)
                    if other != server_index
                },
                "wired_weights": {
                    f"server_{other}": wired_weights[server_index][other]
                    for other in range(num_servers)
                    if other != server_index
                },
                "physical_wired_links": physical_links,
            }
        )
    return servers


# Nested server design (Fig. 3).  When set to an int, build_scheduler_inputs
# builds the network for this many base servers once and then only *adds*
# servers: base servers, IoTs, channels to base servers, expert storage and
# wired links stay identical for every larger server count.
NESTED_SERVER_BASE: int | None = None
NESTED_LINKS_PER_NEW_SERVER = 3
# "mirror_nearest": a new server stores the same experts as the nearest base
# server (an edge replica); "least_replicated": least-replicated experts.
NESTED_NEW_SERVER_EXPERTS = "mirror_nearest"
_NESTED_SALT = 0x2E57ED


def build_balanced_server_positions(num_servers: int, area_size: float) -> list[tuple[float, float]]:
    """round(sqrt(n)) rows, servers split evenly over rows, evenly spaced in a row."""
    rows = max(1, round(math.sqrt(num_servers)))
    base, extra = divmod(num_servers, rows)
    positions: list[tuple[float, float]] = []
    for row in range(rows):
        count = base + (1 if row < extra else 0)
        y = (row + 0.5) * area_size / rows
        for col in range(count):
            positions.append(((col + 0.5) * area_size / count, y))
    return positions


def build_nested_server_positions(
    num_servers: int,
    base_servers: int,
    area_size: float,
    grid: int = 50,
) -> list[tuple[float, float]]:
    """Balanced base layout, then each new server at the inner grid point farthest from all existing ones."""
    positions = build_balanced_server_positions(min(num_servers, base_servers), area_size)
    # Candidates cover the inner 80% of the area so new servers are not put on the border.
    candidates = [
        (area_size * (0.1 + 0.8 * (i + 0.5) / grid), area_size * (0.1 + 0.8 * (j + 0.5) / grid))
        for i in range(grid)
        for j in range(grid)
    ]
    while len(positions) < num_servers:
        positions.append(
            max(candidates, key=lambda point: (min(math.dist(point, q) for q in positions), -point[0], -point[1]))
        )
    return positions


def _routes_from_links(
    num_servers: int,
    links: Mapping[tuple[int, int], tuple[float, float]],
    weight_max: float,
    rate_min: float,
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """Minimum-cost routes (same rule as build_physical_wired_shortest_paths)."""
    route_cost = [[math.inf] * num_servers for _ in range(num_servers)]
    hops = [[math.inf] * num_servers for _ in range(num_servers)]
    bottleneck = [[0.0] * num_servers for _ in range(num_servers)]
    for index in range(num_servers):
        route_cost[index][index] = 0.0
        hops[index][index] = 0
        bottleneck[index][index] = math.inf
    for (left, right), (rate, edge_weight) in links.items():
        route_cost[left][right] = route_cost[right][left] = edge_weight
        hops[left][right] = hops[right][left] = 1
        bottleneck[left][right] = bottleneck[right][left] = rate
    for mid in range(num_servers):
        for src in range(num_servers):
            if route_cost[src][mid] == math.inf:
                continue
            for dst in range(num_servers):
                if route_cost[mid][dst] == math.inf:
                    continue
                candidate_cost = route_cost[src][mid] + route_cost[mid][dst]
                candidate_hops = hops[src][mid] + hops[mid][dst]
                candidate_rate = min(bottleneck[src][mid], bottleneck[mid][dst])
                cost_tied = math.isclose(candidate_cost, route_cost[src][dst], rel_tol=1e-12, abs_tol=1e-12)
                if (
                    candidate_cost < route_cost[src][dst] - 1e-12
                    or (
                        cost_tied
                        and (
                            candidate_rate > bottleneck[src][dst]
                            or (
                                math.isclose(candidate_rate, bottleneck[src][dst])
                                and candidate_hops < hops[src][dst]
                            )
                        )
                    )
                ):
                    route_cost[src][dst] = candidate_cost
                    hops[src][dst] = candidate_hops
                    bottleneck[src][dst] = candidate_rate
    wired_rates: dict[int, dict[int, float]] = {index: {} for index in range(num_servers)}
    wired_weights: dict[int, dict[int, float]] = {index: {} for index in range(num_servers)}
    for src in range(num_servers):
        for dst in range(num_servers):
            if src == dst:
                continue
            wired_rates[src][dst] = bottleneck[src][dst] if bottleneck[src][dst] > 0.0 else rate_min
            wired_weights[src][dst] = float(
                route_cost[src][dst] if route_cost[src][dst] != math.inf else num_servers * weight_max
            )
    return wired_rates, wired_weights


def _build_nested_scheduler_inputs(
    *,
    tasks,
    feature_bits_by_name,
    experts,
    rng: random.Random,
    base_servers: int,
    num_servers: int,
    num_iot_features: int,
    num_iot_devices: int,
    features_per_device_range: tuple[int, int],
    server_feature_overlap_ratio: float,
    global_random_feature_fraction: float,
    experts_per_server,
    server_processing_rate: float,
    server_gpu_memory: float,
    server_gpu_memory_range,
    wired_rate_range,
    wired_extra_link_probability: float,
    wired_edge_weight_range: tuple[float, float],
    max_device_power: float,
    area_size: float,
    cell_radius: float,
    num_antennas: int,
    random_seed: int,
) -> "SchedulerInputs":
    """Base network for `base_servers` servers, then add servers one by one.

    Server k (k >= base_servers) only depends on the seed, k and servers
    0..k-1, so the network for n servers is a superset of the one for any
    smaller n.  New servers: farthest-point position, expert replicas
    (least-replicated experts first), GPU and links to the
    NESTED_LINKS_PER_NEW_SERVER nearest existing servers, all from their own
    RNG.  IoTs keep their positions, features and channels to existing
    servers; IoTs within cell_radius of a new server gain one more reachable
    server with a newly drawn channel.
    """
    base_servers = max(1, int(base_servers))
    positions = build_nested_server_positions(max(num_servers, base_servers), base_servers, area_size)
    base_count = min(num_servers, base_servers)
    servers = build_simple_servers(
        num_servers=base_count,
        experts=experts,
        experts_per_server=experts_per_server,
        processing_rate=server_processing_rate,
        gpu_memory=server_gpu_memory,
        gpu_memory_range=server_gpu_memory_range,
        wired_rate_range=wired_rate_range,
        wired_extra_link_probability=wired_extra_link_probability,
        wired_edge_weight_range=wired_edge_weight_range,
        rng=rng,
        deployment_seed=random_seed,
    )
    devices, topology = build_simple_devices(
        num_iot_features=num_iot_features,
        num_iot_devices=num_iot_devices,
        num_servers=base_count,
        features_per_device_range=features_per_device_range,
        server_feature_overlap_ratio=server_feature_overlap_ratio,
        global_random_feature_fraction=global_random_feature_fraction,
        area_size=area_size,
        cell_radius=cell_radius,
        num_antennas=num_antennas,
        rng=rng,
        max_power=max_device_power,
        server_positions=positions[:base_count],
    )
    if num_servers > base_count:
        expert_ids = sorted(experts, key=lambda item: int(item.rsplit("_", 1)[1]))
        low, high = normalize_experts_per_server(experts_per_server)
        high = min(high, len(expert_ids))
        low = min(low, high)
        gpu_min, gpu_max = normalize_float_range(
            server_gpu_memory_range, (server_gpu_memory, server_gpu_memory), "server_gpu_memory_range"
        )
        rate_min, rate_max = wired_rate_range or (1e9, 1e9)
        weight_min, weight_max = wired_edge_weight_range
        links: dict[tuple[int, int], tuple[float, float]] = {}
        for item in servers[0].get("physical_wired_links", []):
            left = int(str(item["src"]).rsplit("_", 1)[1])
            right = int(str(item["dst"]).rsplit("_", 1)[1])
            links[(min(left, right), max(left, right))] = (float(item["rate"]), float(item["weight"]))
        replica = {expert: 0 for expert in expert_ids}
        for server in servers:
            for expert in server["stored_experts"]:
                replica[expert] += 1
        for k in range(base_count, num_servers):
            krng = random.Random((int(random_seed) * 1_000_003 + k * 7_919) ^ _NESTED_SALT)
            slots = krng.randint(low, high)
            stored: list[str] = []
            if NESTED_NEW_SERVER_EXPERTS == "mirror_nearest":
                mirror = min(range(base_count), key=lambda j: (math.dist(positions[k], positions[j]), j))
                stored = list(servers[mirror]["stored_experts"])
                for expert in stored:
                    replica[expert] += 1
            while NESTED_NEW_SERVER_EXPERTS != "mirror_nearest" and len(stored) < slots:
                candidates = [e for e in expert_ids if e not in stored]
                minimum = min(replica[e] for e in candidates)
                choice = krng.choice([e for e in candidates if replica[e] == minimum])
                stored.append(choice)
                replica[choice] += 1
            stored.sort(key=lambda item: int(item.rsplit("_", 1)[1]))
            gpu = gpu_min if gpu_min == gpu_max else krng.uniform(gpu_min, gpu_max)
            nearest = sorted(range(k), key=lambda j: (math.dist(positions[k], positions[j]), j))
            for j in nearest[:NESTED_LINKS_PER_NEW_SERVER]:
                links[(j, k)] = (krng.uniform(rate_min, rate_max), krng.uniform(weight_min, weight_max))
            servers.append({
                "id": f"server_{k}",
                "processing_rate": server_processing_rate_for(k, server_processing_rate, random_seed),
                "gpu_memory": gpu,
                "stored_experts": stored,
                "active_experts": [],
                "wired_rates": {},
                "wired_weights": {},
                "physical_wired_links": [],
            })
            server_id = f"server_{k}"
            newly_connected = []
            for device in devices:
                x, y = device["position"]
                distance = math.dist((x, y), positions[k])
                if distance > cell_radius:
                    continue
                link_distance = min(max(distance, 10.0), cell_radius)
                vector, gain, phase = channel_from_distance(link_distance, num_antennas, krng)
                device["connected_edge_server_ids"].append(server_id)
                device["channel_gain"][server_id] = gain
                device["channel_vector"][server_id] = vector_to_json(vector)
                device["channel_phase"][server_id] = phase
                device["phase_angle"][server_id] = geometric_angle((x, y), positions[k])
                device["distance"][server_id] = round(float(link_distance), 6)
                newly_connected.append(device)
            for device in newly_connected:
                device["spatial_correlation"][server_id] = {other["id"]: 1.0 for other in newly_connected}
        wired_rates, wired_weights = _routes_from_links(num_servers, links, weight_max, rate_min)
        link_list = [
            {"src": f"server_{left}", "dst": f"server_{right}", "rate": rate, "weight": weight}
            for (left, right), (rate, weight) in sorted(links.items())
        ]
        for index, server in enumerate(servers):
            server["wired_rates"] = {f"server_{o}": wired_rates[index][o] for o in range(num_servers) if o != index}
            server["wired_weights"] = {f"server_{o}": wired_weights[index][o] for o in range(num_servers) if o != index}
            server["physical_wired_links"] = link_list
        topology = dict(topology)
        topology["server_positions"] = {
            f"server_{index}": [round(x, 6), round(y, 6)] for index, (x, y) in enumerate(positions[:num_servers])
        }
        topology["nested_server_base"] = base_count
    return SchedulerInputs(
        tasks=tasks,
        feature_bits_by_name=feature_bits_by_name,
        experts=experts,
        servers=servers,
        devices=devices,
        topology=topology,
    )


def build_server_positions(num_servers: int, area_size: float) -> list[tuple[float, float]]:
    cell_size = math.ceil(math.sqrt(num_servers))
    step = int(area_size) // cell_size
    positions: list[tuple[float, float]] = []
    for row in range(cell_size):
        for col in range(cell_size):
            if len(positions) >= num_servers:
                break
            positions.append((float(row * step + step // 2), float(col * step + step // 2)))
    return positions



def geometric_angle(device_position: tuple[float, float], server_position: tuple[float, float]) -> float:
    """Counterclockwise angle from MEC horizontal axis to the IoT device."""
    angle = math.atan2(
        device_position[1] - server_position[1],
        device_position[0] - server_position[0],
    )
    return angle if angle >= 0.0 else angle + (2.0 * math.pi)

def channel_from_distance(
    distance: float,
    num_antennas: int,
    rng: random.Random,
) -> tuple[list[complex], float, float]:
    """Return scalar channel amplitude and fading phase for one link.

    PL(d) = 50 + 15 log10(d) dB, with a 10 m reference-distance floor.
    Shadowing is a power loss in dB; CN(0,1) fading has unit mean power.
    num_antennas remains metadata: this is not a MIMO channel model.
    """
    del num_antennas
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and non-negative.")
    link_distance = max(distance, 10.0)
    path_loss_db = 50.0 + 15.0 * math.log10(link_distance)
    shadowing_db = rng.gauss(0.0, 1.0)
    power_loss = 10.0 ** ((path_loss_db + shadowing_db) / 10.0)
    fading_std = 1.0 / math.sqrt(2.0)
    small_scale_fading = complex(rng.gauss(0.0, fading_std), rng.gauss(0.0, fading_std))
    channel_gain = abs(small_scale_fading) / math.sqrt(power_loss)
    channel_phase = math.atan2(small_scale_fading.imag, small_scale_fading.real)
    return [], channel_gain, channel_phase

def vector_to_json(vector: Sequence[complex]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in vector]


def spatial_correlation(vector_a: Sequence[complex], vector_b: Sequence[complex]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 1.0
    numerator = abs(sum(a.conjugate() * b for a, b in zip(vector_a, vector_b))) ** 2
    denominator = sum(abs(a) ** 2 for a in vector_a) * sum(abs(b) ** 2 for b in vector_b)
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(float(numerator / denominator), 1.0))




def circular_feature_window(feature_pool: Sequence[int], start: int, size: int) -> list[int]:
    if not feature_pool:
        return []
    size = max(1, min(size, len(feature_pool)))
    return [feature_pool[(start + offset) % len(feature_pool)] for offset in range(size)]


def fill_features_from_pool(
    features: set[int],
    pool: Sequence[int],
    target_count: int,
    rng: random.Random,
) -> None:
    if len(features) >= target_count:
        return
    candidates = [feature for feature in pool if feature not in features]
    if not candidates:
        return
    take_count = min(target_count - len(features), len(candidates))
    features.update(rng.sample(candidates, take_count))

def build_server_feature_pools(
    feature_pool: Sequence[int],
    num_servers: int,
    overlap_ratio: float,
    rng: random.Random,
) -> dict[int, list[int]]:
    """Split global feature indexes into server pools with limited neighbor overlap."""
    if num_servers < 1:
        return {}
    overlap_ratio = min(max(overlap_ratio, 0.0), 1.0)
    chunks = [list(chunk) for chunk in _split_evenly(list(feature_pool), num_servers)]
    avg_chunk_size = max(1, math.ceil(len(feature_pool) / num_servers))
    overlap_count = max(1, round(avg_chunk_size * overlap_ratio)) if overlap_ratio > 0.0 else 0
    pools: dict[int, set[int]] = {index: set(chunks[index]) for index in range(num_servers)}
    for server_index in range(num_servers):
        if overlap_count <= 0:
            continue
        for neighbor_index in ((server_index - 1) % num_servers, (server_index + 1) % num_servers):
            neighbor_chunk = chunks[neighbor_index]
            if neighbor_chunk:
                pools[server_index].update(rng.sample(neighbor_chunk, min(overlap_count, len(neighbor_chunk))))
    return {index: sorted(values) for index, values in pools.items()}


def _split_evenly(values: Sequence[int], parts: int) -> list[list[int]]:
    return [
        list(values[round(index * len(values) / parts): round((index + 1) * len(values) / parts)])
        for index in range(parts)
    ]


def build_position_feature_pools(
    server_feature_pools: Mapping[int, Sequence[int]],
    sector_count: int,
    ring_count: int,
) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    sector_pools: dict[tuple[int, int], list[int]] = {}
    ring_pools: dict[tuple[int, int], list[int]] = {}
    for server_index, pool in server_feature_pools.items():
        pool = list(pool)
        window_size = max(1, math.ceil(len(pool) / max(2, sector_count // 2)))
        for sector_index in range(sector_count):
            sector_pools[(server_index, sector_index)] = circular_feature_window(
                pool,
                sector_index * max(1, window_size // 2),
                window_size,
            )
        ring_window_size = max(1, math.ceil(len(pool) / max(1, ring_count)))
        for ring_index in range(ring_count):
            ring_pools[(server_index, ring_index)] = circular_feature_window(
                pool,
                ring_index * ring_window_size,
                ring_window_size + max(1, ring_window_size // 2),
            )
    return sector_pools, ring_pools


def spatial_feature_set(
    *,
    feature_pool: Sequence[int],
    server_feature_pools: Mapping[int, Sequence[int]],
    sector_feature_pools: Mapping[tuple[int, int], Sequence[int]],
    ring_feature_pools: Mapping[tuple[int, int], Sequence[int]],
    home_server_index: int,
    connected_server_indices: Sequence[int],
    home_distance: float,
    home_angle: float,
    cell_radius: float,
    feature_count: int,
    rng: random.Random,
    global_random_fraction: float = 0.15,
    sector_count: int = 8,
    ring_count: int = 3,
) -> set[int]:
    """Assign spatially correlated feature indexes to one IoT device.

    Nearby IoTs observe largely the same area, so features are drawn first
    from the device's sector pool (direction from its home server), then from
    its ring pool (distance band), then from the home server pool.  A small
    global_random_fraction of features is drawn from the whole feature set.
    Adjacent sector/ring windows overlap, so neighbouring IoTs share most
    features while distant IoTs share few.
    """
    if feature_count <= 0:
        return set()
    feature_count = min(feature_count, len(feature_pool))
    sector_width = (2.0 * math.pi) / max(1, sector_count)
    sector_index = int(home_angle // sector_width) % max(1, sector_count)
    ring_index = min(
        max(1, ring_count) - 1,
        int(max(0.0, home_distance) / max(cell_radius, 1e-9) * max(1, ring_count)),
    )
    if feature_count > 1:
        global_count = min(
            feature_count - 1,
            int(round(feature_count * max(0.0, global_random_fraction))),
        )
    else:
        global_count = 0
    local_target = feature_count - global_count

    features: set[int] = set()
    fill_features_from_pool(
        features, sector_feature_pools.get((home_server_index, sector_index), []), local_target, rng
    )
    fill_features_from_pool(
        features, ring_feature_pools.get((home_server_index, ring_index), []), local_target, rng
    )
    fill_features_from_pool(
        features, server_feature_pools.get(home_server_index, []), local_target, rng
    )
    fill_features_from_pool(features, feature_pool, feature_count, rng)
    return set(sorted(features))


def build_simple_devices(
    num_iot_features: int,
    num_iot_devices: int,
    num_servers: int,
    features_per_device_range: tuple[int, int],
    area_size: float,
    cell_radius: float,
    num_antennas: int,
    rng: random.Random,
    max_power: float = TX_PSD_W_PER_HZ,
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
    server_positions: Sequence[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create IoT devices with scalar amplitude channels and max_power in W/Hz."""
    if num_iot_features < 1:
        raise ValueError("num_iot_features must be at least 1.")
    if num_iot_devices < 1:
        raise ValueError("num_iot_devices must be at least 1.")
    if area_size <= 0 or cell_radius <= 0:
        raise ValueError("area_size and cell_radius must be positive.")
    if num_antennas < 1:
        raise ValueError("num_antennas must be at least 1.")
    min_features, max_features = features_per_device_range
    if min_features < 1 or max_features < min_features:
        raise ValueError("features_per_device_range must be valid and positive.")
    max_features = min(max_features, num_iot_features)

    if server_positions is None:
        server_positions = build_server_positions(num_servers, area_size)
    else:
        server_positions = [tuple(position) for position in server_positions]
    devices: list[dict[str, Any]] = []
    association_map: list[dict[str, Any]] = []
    feature_pool = list(range(num_iot_features))
    sector_count = 8
    ring_count = 3
    server_feature_pools = build_server_feature_pools(
        feature_pool,
        num_servers,
        server_feature_overlap_ratio,
        rng,
    )
    sector_feature_pools, ring_feature_pools = build_position_feature_pools(
        server_feature_pools,
        sector_count,
        ring_count,
    )

    while len(devices) < num_iot_devices:
        x = rng.uniform(0.0, area_size)
        y = rng.uniform(0.0, area_size)
        distances = [
            math.dist((x, y), server_position)
            for server_position in server_positions
        ]
        connected = [
            index
            for index, distance in enumerate(distances)
            if distance <= cell_radius
        ]
        if not connected:
            continue

        device_index = len(devices)
        feature_count = rng.randint(min_features, max_features)

        channel_gain: dict[str, float] = {}
        channel_vector: dict[str, list[list[float]]] = {}
        raw_channel_vectors: dict[str, list[complex]] = {}
        channel_phase: dict[str, float] = {}
        phase_angle: dict[str, float] = {}
        distance_by_server: dict[str, float] = {}
        server_ids = [f"server_{index}" for index in connected]
        for server_index in connected:
            server_id = f"server_{server_index}"
            link_distance = min(max(distances[server_index], 10.0), cell_radius)
            vector, gain, channel_phase_value = channel_from_distance(
                link_distance,
                num_antennas,
                rng,
            )
            channel_gain[server_id] = gain
            raw_channel_vectors[server_id] = vector
            channel_vector[server_id] = vector_to_json(vector)
            channel_phase[server_id] = channel_phase_value
            phase_angle[server_id] = geometric_angle((x, y), server_positions[server_index])
            distance_by_server[server_id] = round(float(link_distance), 6)

        home_server_index = min(connected, key=lambda index: distances[index])
        home_distance = min(max(distances[home_server_index], 10.0), cell_radius)
        home_angle = geometric_angle((x, y), server_positions[home_server_index])
        features = spatial_feature_set(
            feature_pool=feature_pool,
            server_feature_pools=server_feature_pools,
            sector_feature_pools=sector_feature_pools,
            ring_feature_pools=ring_feature_pools,
            home_server_index=home_server_index,
            connected_server_indices=connected,
            home_distance=home_distance,
            home_angle=home_angle,
            cell_radius=cell_radius,
            feature_count=feature_count,
            rng=rng,
            global_random_fraction=global_random_feature_fraction,
            sector_count=sector_count,
            ring_count=ring_count,
        )
        association_map.append({
            "device_id": device_index,
            "connected_edge_server_ids": list(connected),
            "distances": [round(float(distances[index]), 2) for index in connected],
            "angles": [
                round(float(geometric_angle((x, y), server_positions[index])), 6)
                for index in connected
            ],
        })
        devices.append(
            {
                "id": f"device_{device_index}",
                "home_server": f"server_{home_server_index}",
                "connected_edge_server_ids": server_ids,
                "features": [feature_name(index) for index in sorted(features)],
                "position": [round(x, 6), round(y, 6)],
                "channel_gain": channel_gain,
                "channel_vector": channel_vector,
                "channel_phase": channel_phase,
                "_raw_channel_vector": raw_channel_vectors,
                "spatial_correlation": {},
                "phase_angle": phase_angle,
                "distance": distance_by_server,
                "max_power": max_power,
            }
        )

    ensure_iot_feature_coverage(
        devices=devices,
        feature_pool=feature_pool,
        server_feature_pools=server_feature_pools,
        max_features_per_device=max_features,
        rng=rng,
    )

    for server_index in range(num_servers):
        server_id = f"server_{server_index}"
        connected_device_indices = [
            index
            for index, device in enumerate(devices)
            if server_id in device["connected_edge_server_ids"]
        ]
        for index in connected_device_indices:
            device = devices[index]
            correlations = {}
            vector = device["_raw_channel_vector"][server_id]
            for other_index in connected_device_indices:
                other = devices[other_index]
                other_id = other["id"]
                other_vector = other["_raw_channel_vector"][server_id]
                correlations[other_id] = spatial_correlation(vector, other_vector)
            device["spatial_correlation"][server_id] = correlations

    for device in devices:
        device.pop("_raw_channel_vector", None)

    topology = {
        "area_size": area_size,
        "cell_radius": cell_radius,
        "num_antennas": num_antennas,
        "channel_model": "scalar_unit_power_fading_sqrt_power_loss",
        "channel_gain_unit": "amplitude (squared for received power)",
        "device_power_unit": "W/Hz",
        "distance_model": "geometry_based_link_distance_with_10m_floor",
        "angle_model": "geometry_based_counterclockwise_from_server_horizontal_axis",
        "feature_model": "server_pool_position_correlated_with_global_randomness",
        "feature_model_parameters": {
            "server_feature_overlap_ratio": server_feature_overlap_ratio,
            "global_random_feature_fraction": global_random_feature_fraction,
            "sector_count": sector_count,
            "ring_count": ring_count,
            "server_feature_pool_sizes": {
                f"server_{index}": len(pool)
                for index, pool in server_feature_pools.items()
            },
        },
        "association_policy": "all_edge_servers_within_cell_radius",
        "association_map": association_map,
        "server_positions": {
            f"server_{index}": [round(x, 6), round(y, 6)]
            for index, (x, y) in enumerate(server_positions)
        },
    }
    return devices, topology




@dataclass(frozen=True)
class SchedulerInputs:
    tasks: list[dict[str, Any]]
    feature_bits_by_name: dict[str, float]
    experts: dict[str, dict[str, float | int]]
    servers: list[dict[str, Any]]
    devices: list[dict[str, Any]]
    topology: dict[str, Any]


def build_scheduler_inputs(
    graphs: Sequence[nx.DiGraph],
    num_experts: int,
    num_iot_features: int,
    expert_memory_range: tuple[float, float] | None = None,
    expert_memory_sizes_mb: Sequence[float] | None = None,
    expert_inference_times_ms: Sequence[float] | None = None,
    expert_hidden_dimensions: Sequence[int] | None = None,
    expert_parameter_counts: Sequence[int] | None = None,
    inference_unit_cost_per_mb: float = 0.06,
    reasoning_data_sizes_bytes: Sequence[int] = (512,),
    feature_bits_range: tuple[float, float] | None = None,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (5, 12),
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
    experts_per_server: int | Sequence[int] = 4,
    server_processing_rate: float = 300e6,
    server_gpu_memory: float = 8192.0,
    server_gpu_memory_range: tuple[float, float] | None = None,
    wired_rate_range: tuple[float, float] | None = None,
    wired_extra_link_probability: float = 0.05,
    wired_edge_weight_range: tuple[float, float] = (1.0, 3.0),
    default_feature_bits: float = 12000.0,
    max_device_power: float = TX_PSD_W_PER_HZ,
    area_size: float = 1000.0,
    cell_radius: float = 300.0,
    num_antennas: int = 4,
    loss_threshold: float | None = None,
    random_seed: int = 42,
) -> SchedulerInputs:
    """Build the shared task/network objects used by all schedulers."""
    rng = random.Random(random_seed)
    task_rng = random.Random(random_seed ^ 0x5EED5EED)
    tasks = graphs_to_tasks(
        graphs,
        loss_threshold,
        reasoning_data_sizes_bytes=reasoning_data_sizes_bytes,
        default_feature_bits=default_feature_bits,
        rng=task_rng,
    )
    feature_bits_by_name = build_feature_bits(num_iot_features, default_feature_bits, feature_bits_range, rng)
    experts = build_simple_experts(
        num_experts,
        memory_range=expert_memory_range,
        memory_sizes_mb=expert_memory_sizes_mb,
        inference_times_ms=expert_inference_times_ms,
        hidden_dimensions=expert_hidden_dimensions,
        parameter_counts=expert_parameter_counts,
        inference_unit_cost_per_mb=inference_unit_cost_per_mb,
        rng=rng,
    )
    nested_base = NESTED_SERVER_BASE
    if nested_base is not None:
        return _build_nested_scheduler_inputs(
            tasks=tasks,
            feature_bits_by_name=feature_bits_by_name,
            experts=experts,
            rng=rng,
            base_servers=int(nested_base),
            num_servers=num_servers,
            num_iot_features=num_iot_features,
            num_iot_devices=num_iot_devices,
            features_per_device_range=features_per_device_range,
            server_feature_overlap_ratio=server_feature_overlap_ratio,
            global_random_feature_fraction=global_random_feature_fraction,
            experts_per_server=experts_per_server,
            server_processing_rate=server_processing_rate,
            server_gpu_memory=server_gpu_memory,
            server_gpu_memory_range=server_gpu_memory_range,
            wired_rate_range=wired_rate_range,
            wired_extra_link_probability=wired_extra_link_probability,
            wired_edge_weight_range=wired_edge_weight_range,
            max_device_power=max_device_power,
            area_size=area_size,
            cell_radius=cell_radius,
            num_antennas=num_antennas,
            random_seed=random_seed,
        )
    servers = build_simple_servers(
        num_servers=num_servers,
        experts=experts,
        experts_per_server=experts_per_server,
        processing_rate=server_processing_rate,
        gpu_memory=server_gpu_memory,
        gpu_memory_range=server_gpu_memory_range,
        wired_rate_range=wired_rate_range,
        wired_extra_link_probability=wired_extra_link_probability,
        wired_edge_weight_range=wired_edge_weight_range,
        rng=rng,
        deployment_seed=random_seed,
    )
    devices, topology = build_simple_devices(
        num_iot_features=num_iot_features,
        num_iot_devices=num_iot_devices,
        num_servers=num_servers,
        features_per_device_range=features_per_device_range,
        server_feature_overlap_ratio=server_feature_overlap_ratio,
        global_random_feature_fraction=global_random_feature_fraction,
        area_size=area_size,
        cell_radius=cell_radius,
        num_antennas=num_antennas,
        rng=rng,
        max_power=max_device_power,
    )
    return SchedulerInputs(
        tasks=tasks,
        feature_bits_by_name=feature_bits_by_name,
        experts=experts,
        servers=servers,
        devices=devices,
        topology=topology,
    )
