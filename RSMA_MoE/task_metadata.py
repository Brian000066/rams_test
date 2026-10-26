from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import networkx as nx


@dataclass(frozen=True)
class GatingSpec:
    """How per-subtask gating weights and expert confidence are generated.

    For every subtask, K of the global experts receive a raw score
    z ~ U(score_range); all other experts get gating weight 0.  K equals the
    number of experts stored on the subtask's anchor server (10-15 with the
    default deployment); ``num_candidates`` is only used when no deployment
    is given or when it is set explicitly.  The
    gating weights are softmax(z / temperature) over the candidates, so they
    sum to 1.  Confidence is relative to the best candidate:
    pi_p = G_p / max_q G_q (the best candidate has pi = 1, non-candidates 0).

    When ``stored_experts_by_server`` is given, one anchor server is drawn
    uniformly and ``anchor_count`` of its stored experts receive the highest
    raw scores, so the global top-``anchor_count`` experts are always
    co-located on one server.
    """

    num_candidates: Optional[int] = None
    anchor_count: int = 3
    secondary_anchor_count: int = 0
    score_range: tuple[float, float] = (0.1, 1.0)
    temperature: float = 1.0
    stored_experts_by_server: Optional[tuple[tuple[int, ...], ...]] = None


def make_gating_and_confidence(
    num_experts: int,
    rng: random.Random,
    spec: GatingSpec | None = None,
) -> tuple[list[float], list[float]]:
    """Return (gating_weights, expert_confidence) for one subtask."""
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1.")
    spec = spec or GatingSpec()
    low, high = spec.score_range
    if not 0.0 <= low <= high:
        raise ValueError("gating score_range must satisfy 0 <= min <= max.")
    if spec.temperature <= 0.0:
        raise ValueError("gating temperature must be positive.")

    anchored: list[int] = []
    local: list[int] = []
    stored = spec.stored_experts_by_server
    if stored:
        anchor_server = rng.randrange(len(stored))
        local = [index for index in stored[anchor_server] if index < num_experts]
    if spec.num_candidates is not None:
        num_candidates = int(spec.num_candidates)
    elif local:
        # Follow the anchor server: as many scored candidates as it stores.
        num_candidates = len(local)
    else:
        num_candidates = 10
    num_candidates = max(1, min(num_candidates, num_experts))
    if local and spec.anchor_count > 0:
        anchor_count = min(int(spec.anchor_count), len(local), num_candidates)
        anchored = rng.sample(local, anchor_count)
    secondary: list[int] = []
    if stored and len(stored) > 1 and spec.secondary_anchor_count > 0:
        other_servers = [index for index in range(len(stored)) if index != anchor_server]
        second_server = rng.choice(other_servers)
        pool = [index for index in stored[second_server] if index < num_experts and index not in set(anchored)]
        count = min(int(spec.secondary_anchor_count), len(pool), max(0, num_candidates - len(anchored)))
        secondary = rng.sample(pool, count)
    taken = set(anchored) | set(secondary)
    others_pool = [index for index in range(num_experts) if index not in taken]
    others = rng.sample(others_pool, num_candidates - len(anchored) - len(secondary))

    scores = sorted((rng.uniform(low, high) for _ in range(num_candidates)), reverse=True)
    # Anchored experts take the highest scores; the order inside each part is random.
    rng.shuffle(anchored)
    rng.shuffle(secondary)
    rng.shuffle(others)
    # Ranks: primary anchor experts first, then the secondary server's experts.
    candidates = anchored + secondary + others

    max_score = scores[0]
    exp_scores = [math.exp((score - max_score) / spec.temperature) for score in scores]
    total = sum(exp_scores)
    candidate_weights = [value / total for value in exp_scores]

    rounded = [round(weight, 6) for weight in candidate_weights]
    # Put the rounding residual on the top candidate so the sum is exactly 1.
    rounded[0] = round(1.0 - sum(rounded[1:]), 6)
    top_weight = rounded[0]

    gating = [0.0] * num_experts
    confidence = [0.0] * num_experts
    for expert_index, weight in zip(candidates, rounded):
        gating[expert_index] = weight
        confidence[expert_index] = round(min(1.0, weight / top_weight), 6)
    confidence[candidates[0]] = 1.0
    return gating, confidence


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
    gating_spec: GatingSpec | None = None,
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
    gating_weights, expert_confidence = make_gating_and_confidence(
        num_experts,
        rng,
        gating_spec,
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
        "expert_confidence": expert_confidence,
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

    if min_seconds <= 0:
        raise ValueError("Task deadline must be positive.")
    if max_seconds < min_seconds:
        raise ValueError(
            "TASK_DEADLINE_SECONDS_RANGE must be "
            "(min_seconds, max_seconds)."
        )

    if isinstance(min_seconds, int) and isinstance(max_seconds, int):
        # Integer range such as (3, 5): unchanged behaviour (same random stream as before).
        return rng.randint(min_seconds, max_seconds)
    # Float range such as (1.5, 2.5) or (2.0, 3.0): continuous uniform deadline.
    # (Every float range uses uniform, so all x values of Fig. 4 consume the
    # random stream identically and keep the same tasks.)
    return rng.uniform(float(min_seconds), float(max_seconds))
