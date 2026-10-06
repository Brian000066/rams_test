"""GSSGD phase 2 adapted to this project: DBG grouping only.

The original algorithm separates DCG and DBG. This file keeps only DBG:
SPCI ordering + RSMA group-size + distributed beamforming gain constraints.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

from utils.formulation import DeviceId, DeviceSpec, FormulationEvaluator, GroupSpec, ServerId


def phase2_dbg_grouping(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    selected_device_ids: Iterable[DeviceId],
    required_features: Set[str],
    max_group_size: int,
    beamforming_gain_threshold: float = 0.0,
) -> List[Tuple[DeviceId, ...]]:
    """Group selected SIoTs into DBG/RSMA groups."""
    unassigned = sorted(set(selected_device_ids), key=_id_sort_key)
    groups: List[List[DeviceId]] = []

    while unassigned:
        seed = _best_phase_seed(evaluator, server_id, unassigned)
        group = [seed]
        groups.append(group)
        unassigned.remove(seed)

        remaining = list(unassigned)
        while remaining:
            ranked = sorted(
                remaining,
                key=lambda device_id: max(
                    _spci(devices, evaluator, server_id, device_id, existing_group, required_features)
                    for existing_group in groups
                ),
                reverse=True,
            )

            selected = None
            target_group = None
            best_score = None
            for candidate in ranked:
                for existing_group in groups:
                    if len(existing_group) + 1 > max(1, max_group_size):
                        continue
                    proposed = [*existing_group, candidate]
                    if _beamforming_gain(evaluator, server_id, proposed) < beamforming_gain_threshold:
                        continue
                    score = _spci(devices, evaluator, server_id, candidate, existing_group, required_features)
                    item = (score, len(devices[candidate].features & required_features), str(candidate))
                    if best_score is None or item > best_score:
                        best_score = item
                        selected = candidate
                        target_group = existing_group

            if selected is None or best_score[0] <= 0.0:
                new_seed = ranked[0]
                groups.append([new_seed])
                unassigned.remove(new_seed)
                remaining.remove(new_seed)
                continue

            target_group.append(selected)
            unassigned.remove(selected)
            remaining.remove(selected)

    return [tuple(group) for group in groups]


def _best_phase_seed(evaluator: FormulationEvaluator, server_id: ServerId, candidates: Sequence[DeviceId]) -> DeviceId:
    return max(
        candidates,
        key=lambda device_id: (
            _seed_phase_score(evaluator, server_id, device_id, candidates),
            abs(evaluator.channel_gain(device_id, server_id)),
            str(device_id),
        ),
    )


def _seed_phase_score(
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    candidate: DeviceId,
    candidates: Sequence[DeviceId],
) -> float:
    if not candidates:
        return 0.0
    candidate_term = evaluator.db_phase_term(candidate, server_id)
    other_terms = sum(
        (evaluator.db_phase_term(other_id, server_id) for other_id in candidates if other_id != candidate),
        0j,
    )
    return abs(candidate_term + other_terms) / len(candidates)


def _spci(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    candidate: DeviceId,
    group: Sequence[DeviceId],
    required_features: Set[str],
) -> float:
    if not group:
        return 0.0
    proposed = [*group, candidate]
    spatial_phase = _phase_alignment(evaluator, server_id, proposed)
    common_ratio = _common_message_ratio(devices, proposed, set())
    return spatial_phase * common_ratio



def _beamforming_gain(
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    device_ids: Sequence[DeviceId],
) -> float:
    spec = GroupSpec(
        server_id=server_id,
        id="dbg_candidate",
        devices=tuple(device_ids),
        bandwidth=0.0,
        required_features=set(),
    )
    return evaluator.distributed_beamforming_gain(spec)


def _common_message_ratio(
    devices: dict[DeviceId, DeviceSpec],
    device_ids: Sequence[DeviceId],
    required_features: Set[str],
) -> float:
    if not device_ids:
        return 0.0
    feature_sets = [set(devices[device_id].features) for device_id in device_ids]
    if required_features:
        feature_sets = [features & required_features for features in feature_sets]
    union = set().union(*feature_sets)
    if not union:
        return 0.0
    common = set(feature_sets[0])
    for features in feature_sets[1:]:
        common &= features
    return len(common) / len(union)


def _phase_alignment(evaluator: FormulationEvaluator, server_id: ServerId, device_ids: Sequence[DeviceId]) -> float:
    if not device_ids:
        return 0.0
    terms = [evaluator.db_phase_term(device_id, server_id) for device_id in device_ids]
    return abs(sum(terms)) / len(terms)


def _id_sort_key(value: str) -> tuple[str, int, str]:
    prefix, _, suffix = str(value).rpartition("_")
    try:
        return prefix, int(suffix), str(value)
    except ValueError:
        return str(value), 0, str(value)
