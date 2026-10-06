"""GSSGD phase 3 adapted to this project: SER for DBG groups only."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Set, Tuple

from utils.formulation import DeviceId, DeviceSpec, FormulationEvaluator, GroupSpec, ServerId


def phase3_dbg_refinement(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    groups: Sequence[Sequence[DeviceId]],
    unselected_device_ids: Iterable[DeviceId],
    required_features: Set[str],
    beamforming_gain_threshold: float = 0.0,
) -> List[Tuple[DeviceId, ...]]:
    """Run SIoT elimination/replacement for DBG groups while preserving coverage."""
    refined = [list(group) for group in groups if group]
    target_coverage = _groups_features(devices, refined, required_features)

    _eliminate_redundant_devices(
        devices=devices,
        evaluator=evaluator,
        server_id=server_id,
        groups=refined,
        required_features=required_features,
        target_coverage=target_coverage,
    )
    _replace_devices(
        devices=devices,
        evaluator=evaluator,
        server_id=server_id,
        groups=refined,
        unselected=sorted(set(unselected_device_ids), key=_id_sort_key),
        required_features=required_features,
        target_coverage=target_coverage,
        beamforming_gain_threshold=beamforming_gain_threshold,
    )
    return [tuple(group) for group in refined if group and _group_features(devices, group, required_features)]


def _eliminate_redundant_devices(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    groups: List[List[DeviceId]],
    required_features: Set[str],
    target_coverage: Set[str],
) -> None:
    all_devices = sorted({device_id for group in groups for device_id in group}, key=lambda d: len(devices[d].features))
    for device_id in all_devices:
        owner = next((group for group in groups if device_id in group), None)
        if owner is None or len(owner) <= 1:
            continue
        owner.remove(device_id)
        if not _groups_features(devices, groups, required_features).issuperset(target_coverage):
            owner.append(device_id)
            continue


def _replace_devices(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    groups: List[List[DeviceId]],
    unselected: List[DeviceId],
    required_features: Set[str],
    target_coverage: Set[str],
    beamforming_gain_threshold: float,
) -> None:
    for group in groups:
        if not group:
            continue
        improved = True
        while improved:
            improved = False
            current_ratio = _common_message_ratio(devices, group, set())
            current_cost = _group_rsma_cost(devices, evaluator, server_id, group, required_features)
            best = None
            for candidate in list(unselected):
                for replace_count in range(1, len(group) + 1):
                    for replaced in combinations(group, replace_count):
                        replaced_features = set().union(*(devices[d].features & required_features for d in replaced))
                        if not replaced_features.issubset(devices[candidate].features & required_features):
                            continue
                        proposed = [device_id for device_id in group if device_id not in replaced]
                        proposed.append(candidate)
                        if not _groups_coverage_after_replacement(devices, groups, group, proposed, required_features).issuperset(target_coverage):
                            continue
                        if _beamforming_gain(evaluator, server_id, proposed) < beamforming_gain_threshold:
                            continue
                        new_ratio = _common_message_ratio(devices, proposed, set())
                        if new_ratio <= current_ratio:
                            continue
                        new_cost = _group_rsma_cost(devices, evaluator, server_id, proposed, required_features)
                        if new_cost >= current_cost:
                            continue
                        item = (-new_cost, new_ratio, -len(proposed), str(candidate), candidate, tuple(replaced), proposed)
                        if best is None or item > best:
                            best = item
            if best is not None:
                _, _, _, _, candidate, replaced, proposed = best
                group[:] = list(proposed)
                unselected.remove(candidate)
                unselected.extend(replaced)
                improved = True



def _beamforming_gain(
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    group: Sequence[DeviceId],
) -> float:
    spec = GroupSpec(
        server_id=server_id,
        id="ser_db_candidate",
        devices=tuple(group),
        bandwidth=0.0,
        required_features=set(),
    )
    return evaluator.distributed_beamforming_gain(spec)


def _group_rsma_cost(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    group: Sequence[DeviceId],
    required_features: Set[str],
) -> float:
    group_features = _group_features(devices, group, required_features)
    if not group or not group_features:
        return 0.0
    common_power = {device_id: devices[device_id].max_power * evaluator.config.common_power_ratio for device_id in group}
    private_power = {device_id: devices[device_id].max_power * (1.0 - evaluator.config.common_power_ratio) for device_id in group}
    spec = GroupSpec(
        server_id=server_id,
        id="ser_candidate",
        devices=tuple(group),
        bandwidth=0.0,
        required_features=group_features,
        common_power=common_power,
        private_power=private_power,
    )
    bandwidth = evaluator.derive_group_bandwidth_for_budget(spec, evaluator.bandwidth_time_budget())
    coverage_cost = len(group_features) * 0.5
    return bandwidth + coverage_cost

def _groups_coverage_after_replacement(
    devices: dict[DeviceId, DeviceSpec],
    groups: Sequence[Sequence[DeviceId]],
    old_group: Sequence[DeviceId],
    new_group: Sequence[DeviceId],
    required_features: Set[str],
) -> Set[str]:
    coverage: Set[str] = set()
    for group in groups:
        source = new_group if group is old_group else group
        coverage.update(_group_features(devices, source, required_features))
    return coverage



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


def _groups_features(
    devices: dict[DeviceId, DeviceSpec],
    groups: Sequence[Sequence[DeviceId]],
    required_features: Set[str],
) -> Set[str]:
    features: Set[str] = set()
    for group in groups:
        features.update(_group_features(devices, group, required_features))
    return features


def _group_features(
    devices: dict[DeviceId, DeviceSpec],
    group: Sequence[DeviceId],
    required_features: Set[str],
) -> Set[str]:
    features: Set[str] = set()
    for device_id in group:
        features.update(devices[device_id].features & required_features)
    return features


def _id_sort_key(value: str) -> tuple[str, int, str]:
    prefix, _, suffix = str(value).rpartition("_")
    try:
        return prefix, int(suffix), str(value)
    except ValueError:
        return str(value), 0, str(value)
