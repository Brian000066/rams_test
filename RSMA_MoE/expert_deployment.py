"""Task-independent expert deployment shared by DAG generation and schedulers.

The deployment is a pure function of (num_experts, num_servers,
experts_per_server_range, seed).  DAG generation uses it to anchor each
subtask's top gating experts on one server, while every scheduler and the
network-graph export rebuild exactly the same deployment from the same seed.
"""

from __future__ import annotations

import random
from typing import Sequence

DEPLOYMENT_SEED_SALT = 0x5E7ED3B1


def deployment_rng(seed: int) -> random.Random:
    """Return the dedicated RNG used for expert deployment."""
    return random.Random((int(seed) * 1_000_003) ^ DEPLOYMENT_SEED_SALT)


def normalize_experts_per_server(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        low = high = int(value)
    else:
        low, high = (int(item) for item in value)
    if low < 1 or high < low:
        raise ValueError("experts_per_server must be a positive int or (min, max) range.")
    return low, high


# Nested storage design (Fig. 7).  When set to a (min, max) range, every
# deployment is built for this base range first; a larger requested range
# (same width, larger mean) only *adds* replicas to each server, so the base
# deployment (and the gating anchors drawn from it) is kept unchanged.
NESTED_STORAGE_BASE: tuple[int, int] | None = None
_NESTED_STORAGE_SALT = 0x57A6E


def _build_expert_deployment_plain(
    num_experts: int,
    num_servers: int,
    experts_per_server: int | Sequence[int],
    seed: int,
    max_resample_attempts: int = 10_000,
) -> list[list[int]]:
    """Return the expert indexes stored on each server.

    1. Each server draws its number of storage slots from experts_per_server.
       If the total number of slots is smaller than num_experts, all slot
       counts are redrawn so that every expert can be deployed at least once.
    2. Every expert is deployed once (coverage pass), filling servers in
       proportion to their slot counts.
    3. Remaining slots receive replicas, preferring the least-replicated
       experts not already stored on that server.
    """
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1.")
    if num_servers < 1:
        raise ValueError("num_servers must be at least 1.")

    low, high = normalize_experts_per_server(experts_per_server)
    # A server cannot store the same expert twice.
    low = min(low, num_experts)
    high = min(high, num_experts)
    if num_servers * high < num_experts:
        raise ValueError(
            f"{num_servers} servers x at most {high} experts cannot store all "
            f"{num_experts} experts; enlarge experts_per_server or add servers."
        )

    rng = deployment_rng(seed)
    for _ in range(max_resample_attempts):
        slots = [rng.randint(low, high) for _ in range(num_servers)]
        if sum(slots) >= num_experts:
            break
    else:
        raise RuntimeError("Could not draw enough expert slots; check experts_per_server.")

    stored: list[list[int]] = [[] for _ in range(num_servers)]
    replica_count = [0] * num_experts

    # Coverage pass: each expert once, always into the server with the lowest
    # fill ratio (random tie-break) so that small servers are not overfilled.
    coverage_order = list(range(num_experts))
    rng.shuffle(coverage_order)
    for expert_index in coverage_order:
        candidates = [s for s in range(num_servers) if len(stored[s]) < slots[s]]
        fill = min(len(stored[s]) / slots[s] for s in candidates)
        tied = [s for s in candidates if len(stored[s]) / slots[s] == fill]
        server_index = rng.choice(tied)
        stored[server_index].append(expert_index)
        replica_count[expert_index] += 1

    # Replica pass: fill remaining slots with the least-replicated experts.
    for server_index in rng.sample(range(num_servers), num_servers):
        while len(stored[server_index]) < slots[server_index]:
            candidates = [e for e in range(num_experts) if e not in stored[server_index]]
            if not candidates:
                break
            minimum = min(replica_count[e] for e in candidates)
            least = [e for e in candidates if replica_count[e] == minimum]
            expert_index = rng.choice(least)
            stored[server_index].append(expert_index)
            replica_count[expert_index] += 1

    for server_experts in stored:
        server_experts.sort()
    return stored


def build_expert_deployment(
    num_experts: int,
    num_servers: int,
    experts_per_server: int | Sequence[int],
    seed: int,
    max_resample_attempts: int = 10_000,
) -> list[list[int]]:
    """Expert deployment; nested on NESTED_STORAGE_BASE when that is set.

    Without NESTED_STORAGE_BASE this is the plain deployment.  With it, the
    base deployment is built for the base range and each server then gets
    round(mean(requested) - mean(base)) extra replicas, added one per server
    per round (least-replicated expert not yet on that server, ties broken
    by an RNG keyed by (seed, server, round)).  A larger mean therefore gives
    a superset of the smaller deployment on every server.
    """
    base_range = NESTED_STORAGE_BASE
    if base_range is None:
        return _build_expert_deployment_plain(
            num_experts, num_servers, experts_per_server, seed, max_resample_attempts
        )
    stored = _build_expert_deployment_plain(
        num_experts, num_servers, base_range, seed, max_resample_attempts
    )
    low, high = normalize_experts_per_server(experts_per_server)
    base_low, base_high = normalize_experts_per_server(base_range)
    extra = round((low + high) / 2.0 - (base_low + base_high) / 2.0)
    replica_count = [0] * num_experts
    for server_experts in stored:
        for expert_index in server_experts:
            replica_count[expert_index] += 1
    for round_index in range(max(0, extra)):
        for server_index in range(num_servers):
            candidates = [e for e in range(num_experts) if e not in stored[server_index]]
            if not candidates:
                continue
            minimum = min(replica_count[e] for e in candidates)
            least = [e for e in candidates if replica_count[e] == minimum]
            rng = random.Random(
                (int(seed) * 1_000_003 + server_index * 7_919 + round_index * 104_729) ^ _NESTED_STORAGE_SALT
            )
            expert_index = rng.choice(least)
            stored[server_index].append(expert_index)
            replica_count[expert_index] += 1
    for server_experts in stored:
        server_experts.sort()
    return stored
