from __future__ import annotations

import random

import networkx as nx


def make_gating_weights(
    num_experts: int,
    rng: random.Random,
    peak_count_range: tuple[int, int] = (2, 4),
    peak_mass_range: tuple[float, float] = (0.45, 0.60),
) -> list[float]:
    """Create peaked MoE gating weights whose sum is exactly 1.0.

    A small set of experts receives most of the probability mass, while the
    remaining experts share a low tail. This is closer to a real MoE router
    than normalizing independent uniform random numbers.
    """
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1.")
    if num_experts == 1:
        return [1.0]

    min_peaks, max_peaks = peak_count_range
    if min_peaks < 1 or max_peaks < min_peaks:
        raise ValueError("gating_peak_count_range must be (min_count, max_count).")
    min_peaks = min(min_peaks, num_experts)
    max_peaks = min(max_peaks, num_experts)

    min_mass, max_mass = peak_mass_range
    if not 0.0 < min_mass <= max_mass < 1.0:
        raise ValueError("gating_peak_mass_range must satisfy 0 < min <= max < 1.")

    peak_count = rng.randint(min_peaks, max_peaks)
    peak_indices = set(rng.sample(range(num_experts), peak_count))
    peak_mass = rng.uniform(min_mass, max_mass)
    tail_mass = 1.0 - peak_mass

    peak_raw = {index: rng.uniform(0.8, 1.2) for index in peak_indices}
    tail_indices = [index for index in range(num_experts) if index not in peak_indices]
    tail_raw = {index: rng.uniform(0.05, 0.35) for index in tail_indices}

    weights = [0.0 for _ in range(num_experts)]
    peak_total = sum(peak_raw.values())
    tail_total = sum(tail_raw.values())

    for index, value in peak_raw.items():
        weights[index] = peak_mass * value / peak_total
    if tail_indices and tail_total > 0.0:
        for index, value in tail_raw.items():
            weights[index] = tail_mass * value / tail_total

    rounded = [round(weight, 6) for weight in weights]
    rounded[-1] = round(1.0 - sum(rounded[:-1]), 6)
    return rounded


def make_expert_confidence(
    num_experts: int,
    rng: random.Random,
    gating_weights: list[float] | None = None,
) -> list[float]:
    """Create pi(subtask|expert), the confidence of each expert for a node.

    Experts preferred by the gating network should also tend to have higher
    confidence for that subtask; otherwise removing one Top-K expert barely
    changes P(selected experts | subtask).
    """
    if gating_weights is None:
        return [round(rng.uniform(0.55, 0.98), 6) for _ in range(num_experts)]

    if len(gating_weights) != num_experts:
        raise ValueError("gating_weights length must match num_experts.")

    ranked = sorted(range(num_experts), key=lambda index: gating_weights[index], reverse=True)
    # Preserve the original 16-expert suitability proportions when the
    # experiment uses fewer experts: 25% high, 25% medium, and 50% low.
    high_count = max(1, round(num_experts * 0.25))
    medium_count = max(1, round(num_experts * 0.25))
    high_indices = set(ranked[:high_count])
    medium_indices = set(
        ranked[high_count:min(high_count + medium_count, num_experts)]
    )

    confidences: list[float] = []
    for index in range(num_experts):
        if index in high_indices:
            value = rng.uniform(0.85, 0.98)
        elif index in medium_indices:
            value = rng.uniform(0.55, 0.75)
        else:
            value = rng.uniform(0.10, 0.40)
        confidences.append(round(value, 6))
    return confidences


def make_iot_features(
    rng: random.Random,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
) -> list[int]:
    """Choose IoT feature indexes required by one node."""
    min_count, max_count = iot_features_per_node_range

    if min_count < 0:
        raise ValueError("iot_features_per_node_range min cannot be negative.")
    if max_count < min_count:
        raise ValueError(
            "iot_features_per_node_range must be (min_count, max_count)."
        )
    if num_iot_features < 1:
        raise ValueError("num_iot_features must be at least 1.")
    if max_count > num_iot_features:
        raise ValueError(
            "iot_features_per_node_range max cannot exceed num_iot_features."
        )

    feature_count = rng.randint(min_count, max_count)
    return sorted(rng.sample(range(num_iot_features), feature_count))


def make_reconstruction_errors(
    feature_count: int,
    error_range: tuple[float, float],
    rng: random.Random,
) -> list[float]:
    """Create equal per-feature reconstruction errors for one node."""
    low, high = error_range
    if low < 0 or high < low:
        raise ValueError("reconstruction_error_range must be non-negative (min, max).")
    shared_error = round(rng.uniform(low, high), 6)
    return [shared_error for _ in range(feature_count)]
def compute_reconstruction_loss(
    reconstruction_errors: list[float],
    sigma: float,
) -> float:
    """Compute L_rec = 1/(2 sigma^2) * sum_k ||error_k||^2."""
    if sigma <= 0:
        raise ValueError("reconstruction_sigma must be positive.")
    squared_error = sum(error * error for error in reconstruction_errors)
    return round(squared_error / (2.0 * sigma * sigma), 6)


def make_calibration_losses(
    num_samples: int,
    loss_range: tuple[float, float],
    rng: random.Random,
) -> list[float]:
    """Create D_cal node loss samples used to estimate q_hat."""
    if num_samples < 0:
        raise ValueError("num_calibration_samples cannot be negative.")
    low, high = loss_range
    if low < 0 or high < low:
        raise ValueError("calibration_loss_range must be non-negative (min, max).")
    return [round(rng.uniform(low, high), 6) for _ in range(num_samples)]


def make_node_attributes(
    node_name: str,
    layer: int,
    num_experts: int,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
    rng: random.Random,
    reconstruction_sigma: float = 1.0,
    reconstruction_error_range: tuple[float, float] = (0.8, 1.5),
    gating_peak_count_range: tuple[int, int] = (2, 4),
    gating_peak_mass_range: tuple[float, float] = (0.45, 0.60),
    num_calibration_samples: int = 20,
    calibration_loss_range: tuple[float, float] = (1.0, 4.0),
) -> dict[str, object]:
    """Create task-oriented attributes for one node."""
    iot_features = make_iot_features(
        rng=rng,
        num_iot_features=num_iot_features,
        iot_features_per_node_range=iot_features_per_node_range,
    )
    reconstruction_errors = make_reconstruction_errors(
        feature_count=len(iot_features),
        error_range=reconstruction_error_range,
        rng=rng,
    )
    gating_weights = make_gating_weights(
        num_experts,
        rng,
        peak_count_range=gating_peak_count_range,
        peak_mass_range=gating_peak_mass_range,
    )

    return {
        "layer": layer,
        "prompt": f"Complete subtask {node_name} in layer {layer}.",
        "output_key": f"{node_name}_output",
        "required_data": {
            "upstream_outputs": [],
            "iot_features": iot_features,
        },
        "gating_weights": gating_weights,
        "expert_confidence": make_expert_confidence(num_experts, rng, gating_weights),
        "reconstruction_errors": reconstruction_errors,
        "reconstruction_loss": compute_reconstruction_loss(
            reconstruction_errors,
            reconstruction_sigma,
        ),
        "calibration_losses": make_calibration_losses(
            num_samples=num_calibration_samples,
            loss_range=calibration_loss_range,
            rng=rng,
        ),
    }


def update_required_upstream_outputs(graph: nx.DiGraph) -> None:
    """Fill node required_data with outputs from predecessor nodes."""
    for node in graph.nodes:
        predecessor_outputs = [
            {
                "source_node": predecessor,
                "output_key": graph.nodes[predecessor]["output_key"],
            }
            for predecessor in sorted(
                graph.predecessors(node),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
        ]
        graph.nodes[node]["required_data"]["upstream_outputs"] = (
            predecessor_outputs
        )


def validate_num_experts(num_experts: int) -> None:
    """Validate expert count for gating weights."""
    if num_experts < 1:
        raise ValueError("NUM_EXPERTS must be at least 1.")


def make_task_deadline_seconds(
    deadline_seconds_range: tuple[int, int],
    rng: random.Random,
) -> int:
    """Create one task deadline as a relative duration in seconds."""
    min_seconds, max_seconds = deadline_seconds_range

    if min_seconds < 1:
        raise ValueError("Task deadline must be at least 1 second.")
    if max_seconds < min_seconds:
        raise ValueError(
            "TASK_DEADLINE_SECONDS_RANGE must be "
            "(min_seconds, max_seconds)."
        )

    return rng.randint(min_seconds, max_seconds)
