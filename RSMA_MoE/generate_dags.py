"""
Generate layered DAGs and export each one as PNG, GraphML, and JSON.

Install dependencies:
    pip install networkx matplotlib

Run:
    python main.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx

from task_metadata import (
    make_node_attributes,
    make_task_deadline_seconds,
    update_required_upstream_outputs,
    validate_num_experts,
)


@dataclass
class DAGSpec:
    num_nodes: int
    depth: int
    max_width: int
    edge_probability: float = 0.20
    allow_skip_edges: bool = False
    allow_early_branch_end: bool = False
    single_source: bool = True
    single_sink: bool = False
    exact_layer_widths: Optional[list[int]] = None
    num_branches: Optional[int | tuple[int, int]] = None
    branch_length_range: Optional[tuple[int, int]] = None


def build_dag_specs(num_dags: int, common_spec: DAGSpec) -> list[DAGSpec]:
    """Create one shared DAGSpec for each DAG to generate."""
    return [
        DAGSpec(
            num_nodes=common_spec.num_nodes,
            depth=common_spec.depth,
            max_width=common_spec.max_width,
            edge_probability=common_spec.edge_probability,
            allow_skip_edges=common_spec.allow_skip_edges,
            allow_early_branch_end=common_spec.allow_early_branch_end,
            single_source=common_spec.single_source,
            single_sink=common_spec.single_sink,
            exact_layer_widths=(
                list(common_spec.exact_layer_widths)
                if common_spec.exact_layer_widths is not None
                else None
            ),
            num_branches=common_spec.num_branches,
            branch_length_range=common_spec.branch_length_range,
        )
        for _ in range(num_dags)
    ]


def resolve_branch_spec(spec: DAGSpec, rng: random.Random) -> DAGSpec:
    """Return a branch spec whose num_branches is a concrete integer."""
    if spec.num_branches is None or isinstance(spec.num_branches, int):
        return spec

    min_branches, max_branches = spec.num_branches
    if min_branches < 1:
        raise ValueError("num_branches range min must be at least 1.")
    if max_branches < min_branches:
        raise ValueError("num_branches range must be (min_branches, max_branches).")

    feasible: list[int] = []
    endpoint_count = 1 + int(spec.single_sink)
    for branch_count in range(min_branches, max_branches + 1):
        if spec.num_nodes < branch_count + endpoint_count:
            continue
        if spec.branch_length_range is None:
            feasible.append(branch_count)
            continue
        min_length, max_length = spec.branch_length_range
        min_total = endpoint_count + branch_count * min_length
        max_total = endpoint_count + branch_count * max_length
        if min_total <= spec.num_nodes <= max_total:
            feasible.append(branch_count)

    if not feasible:
        raise ValueError(
            f"num_nodes={spec.num_nodes} cannot fit num_branches={spec.num_branches} "
            f"with branch_length_range={spec.branch_length_range}."
        )

    return replace(spec, num_branches=rng.choice(feasible))


def validate_spec(spec: DAGSpec) -> None:
    """Validate DAG generation settings."""
    if spec.num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if spec.depth < 1:
        raise ValueError("depth must be at least 1.")
    if spec.max_width < 1:
        raise ValueError("max_width must be at least 1.")
    if not 0.0 <= spec.edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1.")
    if spec.depth > spec.num_nodes:
        raise ValueError(
            f"depth={spec.depth} cannot be greater than "
            f"num_nodes={spec.num_nodes}."
        )

    if spec.exact_layer_widths is not None:
        widths = spec.exact_layer_widths

        if len(widths) != spec.depth:
            raise ValueError("exact_layer_widths length must equal depth.")
        if sum(widths) != spec.num_nodes:
            raise ValueError("exact_layer_widths sum must equal num_nodes.")
        if any(width < 1 for width in widths):
            raise ValueError("Each layer width must be at least 1.")
        if any(width > spec.max_width for width in widths):
            raise ValueError("No layer width can be greater than max_width.")
        if spec.single_source and widths[0] != 1:
            raise ValueError(
                "single_source=True requires the first layer width to be 1."
            )
        if spec.single_sink and widths[-1] != 1:
            raise ValueError(
                "single_sink=True requires the last layer width to be 1."
            )
        return

    capacity = spec.depth * spec.max_width
    if spec.single_source:
        capacity -= spec.max_width - 1
    if spec.single_sink and spec.depth > 1:
        capacity -= spec.max_width - 1

    if spec.num_nodes > capacity:
        raise ValueError(
            f"These settings can hold at most {capacity} nodes, "
            f"but num_nodes={spec.num_nodes}. Increase depth or max_width."
        )


def allocate_layer_widths(spec: DAGSpec, rng: random.Random) -> list[int]:
    """Distribute num_nodes across depth layers."""
    validate_spec(spec)

    if spec.exact_layer_widths is not None:
        return list(spec.exact_layer_widths)

    widths = [1] * spec.depth
    remaining = spec.num_nodes - spec.depth

    locked_layers: set[int] = set()
    if spec.single_source:
        locked_layers.add(0)
    if spec.single_sink:
        locked_layers.add(spec.depth - 1)

    available_layers = [
        layer for layer in range(spec.depth) if layer not in locked_layers
    ]

    while remaining > 0:
        candidates = [
            layer
            for layer in available_layers
            if widths[layer] < spec.max_width
        ]
        if not candidates:
            raise ValueError("Cannot fit num_nodes with this depth/max_width.")

        minimum_width = min(widths[layer] for layer in candidates)
        narrowest_layers = [
            layer for layer in candidates if widths[layer] == minimum_width
        ]
        chosen_layer = rng.choice(narrowest_layers)
        widths[chosen_layer] += 1
        remaining -= 1

    return widths


def make_layered_nodes(
    graph_index: int,
    layer_widths: Sequence[int],
) -> tuple[list[list[str]], dict[str, int]]:
    """Create node names grouped by layer."""
    layers: list[list[str]] = []
    node_to_layer: dict[str, int] = {}

    node_counter = 0
    for layer_index, width in enumerate(layer_widths):
        current_layer: list[str] = []

        for _ in range(width):
            node_name = f"v{graph_index}_{node_counter}"
            current_layer.append(node_name)
            node_to_layer[node_name] = layer_index
            node_counter += 1

        layers.append(current_layer)

    return layers, node_to_layer



def validate_branch_spec(spec: DAGSpec) -> None:
    """Validate branch-based DAG settings."""
    if spec.num_branches is None:
        return
    if spec.num_branches < 1:
        raise ValueError("num_branches must be at least 1.")
    endpoint_count = 1 + int(spec.single_sink)
    if spec.num_nodes < spec.num_branches + endpoint_count:
        raise ValueError(
            "num_nodes must fit one source, one node per branch, and the "
            "optional shared sink when using branch-based generation."
        )

    if spec.branch_length_range is not None:
        min_length, max_length = spec.branch_length_range
        if min_length < 1:
            raise ValueError("branch_length_range min must be at least 1.")
        if max_length < min_length:
            raise ValueError("branch_length_range must be (min_length, max_length).")
        min_total = endpoint_count + spec.num_branches * min_length
        max_total = endpoint_count + spec.num_branches * max_length
        if not min_total <= spec.num_nodes <= max_total:
            raise ValueError(
                f"num_nodes={spec.num_nodes} cannot fit "
                f"num_branches={spec.num_branches} with "
                f"branch_length_range={spec.branch_length_range}."
            )


def allocate_branch_lengths(spec: DAGSpec, rng: random.Random) -> list[int]:
    """Allocate variable branch lengths while preserving total node count."""
    validate_branch_spec(spec)
    if spec.num_branches is None:
        raise ValueError("num_branches is required for branch allocation.")

    reserved_nodes = 1 + int(spec.single_sink)
    if spec.branch_length_range is None:
        min_length = 1
        max_length = spec.num_nodes - reserved_nodes
    else:
        min_length, max_length = spec.branch_length_range

    lengths = [min_length] * spec.num_branches
    remaining = spec.num_nodes - reserved_nodes - sum(lengths)

    while remaining > 0:
        candidates = [
            branch_index
            for branch_index, length in enumerate(lengths)
            if length < max_length
        ]
        if not candidates:
            raise ValueError("Cannot allocate branch lengths with these settings.")
        branch_index = rng.choice(candidates)
        lengths[branch_index] += 1
        remaining -= 1

    return lengths


def branch_layer_widths(
    branch_lengths: Sequence[int],
    single_sink: bool = False,
) -> list[int]:
    """Return layer widths for one source plus branch positions."""
    max_length = max(branch_lengths)
    widths = [1] + [
        sum(length >= position for length in branch_lengths)
        for position in range(1, max_length + 1)
    ]
    if single_sink:
        widths.append(1)
    return widths


def generate_branch_dag(
    graph_index: int,
    spec: DAGSpec,
    rng: random.Random,
    num_experts: int,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
    reconstruction_sigma: float,
    reconstruction_error_range: tuple[float, float],
    gating_peak_count_range: tuple[int, int],
    gating_peak_mass_range: tuple[float, float],
    num_calibration_samples: int,
    calibration_loss_range: tuple[float, float],
    deadline_seconds: int,
) -> nx.DiGraph:
    """Generate one DAG with adjustable variable-length branches."""
    validate_num_experts(num_experts)
    branch_lengths = allocate_branch_lengths(spec, rng)
    layer_widths = branch_layer_widths(
        branch_lengths,
        single_sink=spec.single_sink,
    )

    graph = nx.DiGraph(
        graph_index=graph_index,
        generation_mode="branches",
        num_nodes=spec.num_nodes,
        num_branches=spec.num_branches,
        branch_lengths=branch_lengths,
        depth=len(layer_widths),
        max_width=max(layer_widths),
        edge_probability=spec.edge_probability,
        allow_skip_edges=spec.allow_skip_edges,
        allow_early_branch_end=True,
        single_source=True,
        single_sink=spec.single_sink,
        task_id=f"task_{graph_index}",
        task_name=f"Task {graph_index}",
        deadline_seconds=deadline_seconds,
        num_experts=num_experts,
        layer_widths=layer_widths,
    )

    source = f"v{graph_index}_0"
    graph.add_node(
        source,
        **make_node_attributes(
            node_name=source,
            layer=0,
            num_experts=num_experts,
            num_iot_features=num_iot_features,
            iot_features_per_node_range=iot_features_per_node_range,
            rng=rng,
            reconstruction_sigma=reconstruction_sigma,
            reconstruction_error_range=reconstruction_error_range,
            gating_peak_count_range=gating_peak_count_range,
            gating_peak_mass_range=gating_peak_mass_range,
            num_calibration_samples=num_calibration_samples,
            calibration_loss_range=calibration_loss_range,
        ),
    )
    graph.nodes[source]["branch_id"] = -1
    graph.nodes[source]["branch_position"] = 0

    node_counter = 1
    branches: list[list[str]] = []
    for branch_id, branch_length in enumerate(branch_lengths):
        branch_nodes: list[str] = []
        previous_node = source
        for branch_position in range(1, branch_length + 1):
            node = f"v{graph_index}_{node_counter}"
            node_counter += 1
            graph.add_node(
                node,
                **make_node_attributes(
                    node_name=node,
                    layer=branch_position,
                    num_experts=num_experts,
                    num_iot_features=num_iot_features,
                    iot_features_per_node_range=iot_features_per_node_range,
                    rng=rng,
                    reconstruction_sigma=reconstruction_sigma,
                    reconstruction_error_range=reconstruction_error_range,
                    gating_peak_count_range=gating_peak_count_range,
                    gating_peak_mass_range=gating_peak_mass_range,
                    num_calibration_samples=num_calibration_samples,
                    calibration_loss_range=calibration_loss_range,
                ),
            )
            graph.nodes[node]["branch_id"] = branch_id
            graph.nodes[node]["branch_position"] = branch_position
            graph.add_edge(previous_node, node)
            previous_node = node
            branch_nodes.append(node)
        branches.append(branch_nodes)

    for source_branch_id, source_branch in enumerate(branches):
        for target_branch_id, target_branch in enumerate(branches):
            if source_branch_id == target_branch_id:
                continue
            for source_node in source_branch:
                source_position = int(graph.nodes[source_node]["branch_position"])
                for target_node in target_branch:
                    target_position = int(graph.nodes[target_node]["branch_position"])
                    if target_position <= source_position:
                        continue
                    if graph.has_edge(source_node, target_node):
                        continue
                    if rng.random() < spec.edge_probability:
                        graph.add_edge(source_node, target_node)

    if spec.single_sink:
        sink = f"v{graph_index}_{node_counter}"
        sink_layer = max(branch_lengths) + 1
        graph.add_node(
            sink,
            **make_node_attributes(
                node_name=sink,
                layer=sink_layer,
                num_experts=num_experts,
                num_iot_features=num_iot_features,
                iot_features_per_node_range=iot_features_per_node_range,
                rng=rng,
                reconstruction_sigma=reconstruction_sigma,
                reconstruction_error_range=reconstruction_error_range,
                gating_peak_count_range=gating_peak_count_range,
                gating_peak_mass_range=gating_peak_mass_range,
                num_calibration_samples=num_calibration_samples,
                calibration_loss_range=calibration_loss_range,
            ),
        )
        graph.nodes[sink]["branch_id"] = -2
        graph.nodes[sink]["branch_position"] = sink_layer
        existing_leaves = [
            node
            for node in graph.nodes
            if node != sink and graph.out_degree(node) == 0
        ]
        for leaf in existing_leaves:
            graph.add_edge(leaf, sink)

    update_required_upstream_outputs(graph)

    if len(graph.nodes) != spec.num_nodes:
        raise RuntimeError(
            f"Generated branch graph has {len(graph.nodes)} nodes; "
            f"expected {spec.num_nodes}."
        )
    sources = [node for node in graph.nodes if graph.in_degree(node) == 0]
    sinks = [node for node in graph.nodes if graph.out_degree(node) == 0]
    if spec.single_source and len(sources) != 1:
        raise RuntimeError(
            f"Generated branch graph has {len(sources)} sources; expected exactly one."
        )
    if spec.single_sink and len(sinks) != 1:
        raise RuntimeError(
            f"Generated branch graph has {len(sinks)} sinks; expected exactly one."
        )

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Generated graph is not a DAG.")

    return graph

def generate_dag(
    graph_index: int,
    spec: DAGSpec,
    rng: random.Random,
    num_experts: int,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
    reconstruction_sigma: float,
    reconstruction_error_range: tuple[float, float],
    gating_peak_count_range: tuple[int, int],
    gating_peak_mass_range: tuple[float, float],
    num_calibration_samples: int,
    calibration_loss_range: tuple[float, float],
    deadline_seconds: int,
) -> nx.DiGraph:
    """Generate one directed acyclic graph."""
    if spec.num_branches is not None:
        branch_spec = resolve_branch_spec(spec, rng)
        return generate_branch_dag(
            graph_index=graph_index,
            spec=branch_spec,
            rng=rng,
            num_experts=num_experts,
            num_iot_features=num_iot_features,
            iot_features_per_node_range=iot_features_per_node_range,
            reconstruction_sigma=reconstruction_sigma,
            reconstruction_error_range=reconstruction_error_range,
            gating_peak_count_range=gating_peak_count_range,
            gating_peak_mass_range=gating_peak_mass_range,
            num_calibration_samples=num_calibration_samples,
            calibration_loss_range=calibration_loss_range,
            deadline_seconds=deadline_seconds,
        )

    validate_num_experts(num_experts)
    layer_widths = allocate_layer_widths(spec, rng)
    layers, node_to_layer = make_layered_nodes(
        graph_index=graph_index,
        layer_widths=layer_widths,
    )

    graph = nx.DiGraph(
        graph_index=graph_index,
        num_nodes=spec.num_nodes,
        depth=spec.depth,
        max_width=spec.max_width,
        edge_probability=spec.edge_probability,
        allow_skip_edges=spec.allow_skip_edges,
        allow_early_branch_end=spec.allow_early_branch_end,
        single_source=spec.single_source,
        single_sink=spec.single_sink,
        task_id=f"task_{graph_index}",
        task_name=f"Task {graph_index}",
        deadline_seconds=deadline_seconds,
        num_experts=num_experts,
        layer_widths=layer_widths,
    )

    for node, layer in node_to_layer.items():
        graph.add_node(
            node,
            **make_node_attributes(
                node_name=node,
                layer=layer,
                num_experts=num_experts,
                num_iot_features=num_iot_features,
                iot_features_per_node_range=iot_features_per_node_range,
                rng=rng,
                reconstruction_sigma=reconstruction_sigma,
                reconstruction_error_range=reconstruction_error_range,
                gating_peak_count_range=gating_peak_count_range,
                gating_peak_mass_range=gating_peak_mass_range,
                num_calibration_samples=num_calibration_samples,
                calibration_loss_range=calibration_loss_range,
            ),
        )

    # Add mandatory edges between adjacent layers so each non-source has a
    # parent. If allow_early_branch_end is false, each non-sink also gets a
    # child; otherwise, some branches may end before the deepest layer.
    for layer_index in range(spec.depth - 1):
        current_layer = layers[layer_index]
        next_layer = layers[layer_index + 1]

        for child_index, child in enumerate(next_layer):
            parent = current_layer[child_index % len(current_layer)]
            graph.add_edge(parent, child)

        if not spec.allow_early_branch_end:
            for parent_index, parent in enumerate(current_layer):
                if graph.out_degree(parent) == 0:
                    child = next_layer[parent_index % len(next_layer)]
                    graph.add_edge(parent, child)

    # Add optional random edges.
    for source_layer in range(spec.depth - 1):
        if spec.allow_skip_edges:
            possible_target_layers = range(source_layer + 1, spec.depth)
        else:
            possible_target_layers = [source_layer + 1]

        for target_layer in possible_target_layers:
            for source in layers[source_layer]:
                for target in layers[target_layer]:
                    if graph.has_edge(source, target):
                        continue
                    if rng.random() < spec.edge_probability:
                        graph.add_edge(source, target)

    update_required_upstream_outputs(graph)

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Generated graph is not a DAG.")

    return graph


def layered_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Calculate stable positions for a layered graph drawing."""
    layer_to_nodes: dict[int, list[str]] = {}

    for node, data in graph.nodes(data=True):
        layer = int(data["layer"])
        layer_to_nodes.setdefault(layer, []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    max_layer_width = max(len(nodes) for nodes in layer_to_nodes.values())

    for layer, nodes in sorted(layer_to_nodes.items()):
        nodes = sorted(nodes, key=lambda name: int(name.rsplit("_", 1)[1]))
        width = len(nodes)

        if width == 1:
            x_values = [0.0]
        else:
            x_values = [index - (width - 1) / 2 for index in range(width)]

        if width > 1 and max_layer_width > 1:
            scale = (max_layer_width - 1) / (width - 1)
            x_values = [x * scale for x in x_values]

        for node, x_value in zip(nodes, x_values):
            positions[node] = (x_value, -float(layer))

    return positions


def visualize_dag(
    graph: nx.DiGraph,
    output_path: Path,
    dpi: int = 180,
) -> None:
    """Save one DAG as a PNG image."""
    positions = layered_positions(graph)

    depth = int(graph.graph["depth"])
    layer_widths = list(graph.graph["layer_widths"])
    widest_layer = max(layer_widths)

    figure_width = max(7.0, widest_layer * 1.7)
    figure_height = max(5.0, depth * 1.8)

    plt.figure(figsize=(figure_width, figure_height))

    nx.draw_networkx(
        graph,
        pos=positions,
        with_labels=True,
        arrows=True,
        node_size=1500,
        font_size=9,
        width=1.2,
        arrowsize=18,
        connectionstyle="arc3,rad=0.03",
    )

    graph_index = graph.graph["graph_index"]
    plt.title(
        f"DAG {graph_index}: "
        f"nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, "
        f"depth={depth}, "
        f"layer widths={layer_widths}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_dag_json(graph: nx.DiGraph, output_path: Path) -> None:
    """Save one DAG as JSON."""
    data = {
        "graph": dict(graph.graph),
        "nodes": [
            {
                "id": node,
                **attributes,
            }
            for node, attributes in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
            }
            for source, target in graph.edges()
        ],
        "topological_order": list(nx.topological_sort(graph)),
    }

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def apply_unique_subtask_templates(
    graph: nx.DiGraph,
    templates: dict[str, dict[str, object]],
) -> None:
    """Reuse unique-subtask metadata across task graph instances."""
    for node in sorted(graph.nodes, key=lambda name: int(str(name).rsplit("_", 1)[1])):
        local_index = str(node).rsplit("_", 1)[-1]
        unique_id = f"subtask_{local_index}"
        attributes = graph.nodes[node]
        required_data = dict(attributes.get("required_data", {}))
        upstream_outputs = list(required_data.get("upstream_outputs", []))

        if unique_id not in templates:
            templates[unique_id] = {
                "iot_features": list(required_data.get("iot_features", [])),
                "gating_weights": list(attributes.get("gating_weights", [])),
                "expert_confidence": list(attributes.get("expert_confidence", [])),
                "reconstruction_errors": list(attributes.get("reconstruction_errors", [])),
                "reconstruction_loss": float(attributes.get("reconstruction_loss", 0.0)),
                "calibration_losses": list(attributes.get("calibration_losses", [])),
            }

        template = templates[unique_id]
        required_data["iot_features"] = list(template["iot_features"])
        required_data["upstream_outputs"] = upstream_outputs
        attributes["required_data"] = required_data
        attributes["unique_subtask_id"] = unique_id
        attributes["gating_weights"] = list(template["gating_weights"])
        attributes["expert_confidence"] = list(template["expert_confidence"])
        attributes["reconstruction_errors"] = list(template["reconstruction_errors"])
        attributes["reconstruction_loss"] = float(template["reconstruction_loss"])
        attributes["calibration_losses"] = list(template["calibration_losses"])


def generate_multiple_dags(
    specs: Sequence[DAGSpec],
    output_dir: Path,
    seed: int = 42,
    image_dpi: int = 180,
    num_experts: int = 1,
    num_iot_features: int = 1,
    iot_features_per_node_range: tuple[int, int] = (1, 1),
    reconstruction_sigma: float = 1.0,
    reconstruction_error_range: tuple[float, float] = (0.8, 1.5),
    gating_peak_count_range: tuple[int, int] = (2, 4),
    gating_peak_mass_range: tuple[float, float] = (0.45, 0.60),
    num_calibration_samples: int = 20,
    calibration_loss_range: tuple[float, float] = (1.0, 4.0),
    task_deadline_seconds_range: tuple[int, int] = (60, 60),
    verbose: bool = True,
    export_artifacts: bool = True,
) -> list[nx.DiGraph]:
    """Generate and export multiple DAGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    graphs: list[nx.DiGraph] = []
    unique_subtask_templates: dict[str, dict[str, object]] = {}

    for graph_index, spec in enumerate(specs, start=1):
        deadline_seconds = make_task_deadline_seconds(
            task_deadline_seconds_range,
            rng,
        )
        graph = generate_dag(
            graph_index=graph_index,
            spec=spec,
            rng=rng,
            num_experts=num_experts,
            num_iot_features=num_iot_features,
            iot_features_per_node_range=iot_features_per_node_range,
            reconstruction_sigma=reconstruction_sigma,
            reconstruction_error_range=reconstruction_error_range,
            gating_peak_count_range=gating_peak_count_range,
            gating_peak_mass_range=gating_peak_mass_range,
            num_calibration_samples=num_calibration_samples,
            calibration_loss_range=calibration_loss_range,
            deadline_seconds=deadline_seconds,
        )
        apply_unique_subtask_templates(graph, unique_subtask_templates)
        graphs.append(graph)

        if not export_artifacts:
            continue

        image_path = output_dir / f"dag_{graph_index}.png"
        graphml_path = output_dir / f"dag_{graph_index}.graphml"
        json_path = output_dir / f"dag_{graph_index}.json"

        visualize_dag(
            graph=graph,
            output_path=image_path,
            dpi=image_dpi,
        )

        graphml_graph = graph.copy()
        for key, value in list(graphml_graph.graph.items()):
            if isinstance(value, (list, dict, tuple, set)):
                graphml_graph.graph[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )
        for _, attributes in graphml_graph.nodes(data=True):
            for key, value in list(attributes.items()):
                if isinstance(value, (list, dict, tuple, set)):
                    attributes[key] = json.dumps(
                        value,
                        ensure_ascii=False,
                    )
        nx.write_graphml(graphml_graph, graphml_path)

        save_dag_json(graph, json_path)

        if not verbose:
            continue

        branch_info = ""
        if graph.graph.get("generation_mode") == "branches":
            branch_info = f", branch_lengths={graph.graph['branch_lengths']}"
        print(
            f"[DAG {graph_index}] "
            f"nodes={graph.number_of_nodes()}, "
            f"edges={graph.number_of_edges()}, "
            f"deadline_seconds={deadline_seconds}, "
            f"layer_widths={graph.graph['layer_widths']}"
            f"{branch_info}"
        )
        print(f"  PNG:     {image_path}")
        print(f"  GraphML: {graphml_path}")
        print(f"  JSON:    {json_path}")

    return graphs
