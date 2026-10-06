from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_location_aware_backhaul import (
    ChainedBackhaulDecision,
    ChainedPipelineResult,
)
from utils.formulation import (
    AssignmentKey,
    DeviceId,
    FormulationEvaluator,
    GroupSpec,
    ServerId,
)


Demand = Tuple[ServerId, str]


class Phase2GroupingBackhaulMixin:
    """Phase II: CFRG IoT selection, compatibility ordering, and grouping."""

    def _run_phase2_for_assignments(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        activated,
        used_memory,
    ) -> ChainedPipelineResult:
        del selected_experts_by_key, activated, used_memory
        assignments = {key: list(value) for key, value in assignments.items()}
        selected_probability = dict(selected_probability)
        local_violations: list[str] = []

        initial_deficits = self._cfrg_feature_deficits(assignments)
        selected_devices, ownership, remaining, selection_report = self._cfrg_select_iots(
            initial_deficits
        )
        for server_id, features in remaining.items():
            for feature in sorted(features):
                local_violations.append(
                    f"CFRG infeasible: feature {feature} required by {server_id} "
                    "has no selectable IoT source"
                )

        order, mst_report = self._cfrg_mst_order(selected_devices)
        self._cfrg_assignments = assignments
        self._cfrg_latest_start = self.evaluator.latest_subtask_start_times(assignments)
        groups, group_records, dp_report = self._cfrg_interval_groups(order, ownership)
        backhaul, backhaul_plan = self._cfrg_backhaul_plan(group_records)
        subtask_features = self._cfrg_subtask_features(assignments, ownership)

        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                servers = self.evaluator.participating_servers(assignments, key)
                server_id = servers[0] if servers else None
                selected = (
                    set()
                    if server_id is None
                    else set(subtask_features.get(self._server_feature_key(key, server_id), set()))
                )
                probability = selected_probability.get(label, 0.0)
                rec_loss = self.evaluator.reconstruction_loss_for_features(subtask, selected)
                loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                    probability,
                    rec_loss,
                )
                threshold = self.evaluator.conformal_loss_threshold(subtask)
                if loss > threshold + 1e-12:
                    local_violations.append(
                        f"CFRG loss: {key} loss {loss:.6g} > threshold {threshold:.6g}"
                    )
                required_probability[label] = self.evaluator.required_selection_probability(
                    subtask,
                    selected,
                )
                reconstruction_loss[label] = rec_loss
                performance_loss[label] = loss

        data_req = self._subtask_data_requirements(assignments, subtask_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(groups)
        resolved_groups, budgets = self._derive_group_bandwidths(
            groups,
            assignments,
            data_req,
            backhaul,
            feature_to_groups,
        )
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(
            self.servers,
            self.devices,
            self.experts,
            self.tasks,
            eval_config,
        )
        evaluation = evaluator.evaluate(
            assignments,
            resolved_groups,
            backhaul,
            subtask_features=subtask_features,
        )
        violations = [*self.scheduler_violations, *local_violations, *evaluation.violations]

        self.phase2_report.update(
            {
                "mode": "CFRG_updated",
                "selection_version": "two_stage_coverage_then_localization",
                "initial_feature_deficits": {
                    server_id: sorted(features)
                    for server_id, features in initial_deficits.items()
                },
                "selected_iots": list(selected_devices),
                "iot_selection": selection_report,
                "feature_ownership": {
                    f"{server_id}:{feature}": device_id
                    for (server_id, feature), device_id in sorted(ownership.items())
                },
                "remaining_feature_deficits": {
                    server_id: sorted(features)
                    for server_id, features in remaining.items()
                    if features
                },
                "mst_order": order,
                "mst": mst_report,
                "dp": dp_report,
                "groups": group_records,
                "backhaul_plan": [decision.__dict__ for decision in backhaul_plan],
                "selection_index_formula": (
                    "Delta_n*|h[n,o_n]|^2 / "
                    "([1+c_bw*b(n)+c_fwd*sum_s omega_net[o_n,s]]*(1+delta_n))"
                ),
                "localization_index_formula": (
                    "normalized_hop_reduction / "
                    "(normalized_c_bw_b(n)*(1+normalized_delta_n)+epsilon)"
                ),
                "localization_acceptance_threshold": 1.0,
                "pruning_rule": "remove selected IoTs whose destination set is empty",
                "compatibility_formula": (
                    "min_s distance(n,m)*(2-cos(phase_angle[n,s]-phase_angle[m,s]))"
                ),
                "utility_formula": (
                    "c_bw*(sum singleton bandwidth-group bandwidth)"
                    "-c_fwd*sum destination weights"
                ),
            }
        )

        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment=assignments,
            subtask_data_requirements=data_req,
            server_required_features={
                server_id: sorted(features)
                for server_id, features in server_features.items()
            },
            rsma_groups=self._group_map(resolved_groups),
            group_required_features={
                self._group_label(group): sorted(group.required_features)
                for group in resolved_groups
            },
            group_time_budgets={
                self._group_key_label(key): value
                for key, value in budgets.items()
            },
            rsma_group_bandwidths=self._group_bandwidth_map(resolved_groups),
            backhaul_plan=backhaul_plan,
            selected_probability=selected_probability,
            required_probability=required_probability,
            reconstruction_loss=reconstruction_loss,
            performance_loss=performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=violations,
            evaluation=evaluation,
        )

    def _cfrg_feature_deficits(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, str]]],
    ) -> Dict[ServerId, Set[str]]:
        deficits: Dict[ServerId, Set[str]] = {
            server_id: set() for server_id in self.servers
        }
        subtask_by_key = {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }
        for key, pairs in assignments.items():
            subtask = subtask_by_key.get(key)
            if subtask is None:
                continue
            for server_id in {server_id for server_id, _ in pairs}:
                deficits.setdefault(server_id, set()).update(subtask.required_features)
        return deficits

    def _cfrg_select_iots(
        self,
        deficits: Mapping[ServerId, Set[str]],
    ) -> tuple[
        list[DeviceId],
        Dict[Demand, DeviceId],
        Dict[ServerId, Set[str]],
        list[dict[str, Any]],
    ]:
        remaining = {
            server_id: set(features)
            for server_id, features in deficits.items()
        }
        selected: list[DeviceId] = []
        ownership: Dict[Demand, DeviceId] = {}
        report: list[dict[str, Any]] = []

        # Stage 1: cover every outstanding (server, feature) demand while
        # charging both the candidate uplink bandwidth and forwarding hops.
        coverage_iteration = 0
        while any(remaining.values()):
            candidates: list[dict[str, Any]] = []
            for device_id in self.devices:
                if device_id in selected:
                    continue
                device = self.devices[device_id]
                covered_by_server = {
                    server_id: set(device.features) & features
                    for server_id, features in remaining.items()
                    if set(device.features) & features
                }
                coverage_gain = sum(len(features) for features in covered_by_server.values())
                if coverage_gain <= 0:
                    continue
                origin = self._updated_best_origin(device_id)
                if origin is None:
                    continue
                required_features = set().union(*covered_by_server.values())
                singleton_bandwidth = self._updated_singleton_bandwidth(
                    device_id,
                    origin,
                    required_features,
                )
                backhaul_hops = sum(
                    self.evaluator.wired_weight(origin, server_id)
                    for server_id in covered_by_server
                )
                incompatibility = self._cfrg_selected_incompatibility(device_id, selected)
                channel_quality = self.evaluator.channel_gain(device_id, origin) ** 2
                denominator = (
                    1.0
                    + self.config.c_bw * singleton_bandwidth
                    + self.config.c_fwd * backhaul_hops
                ) * (1.0 + incompatibility)
                selection_index = self._updated_safe_ratio(
                    coverage_gain * channel_quality,
                    denominator,
                )
                candidates.append(
                    {
                        "device": device_id,
                        "origin": origin,
                        "covered_by_server": covered_by_server,
                        "coverage_gain": coverage_gain,
                        "channel_quality": channel_quality,
                        "singleton_bandwidth": singleton_bandwidth,
                        "backhaul_hops": backhaul_hops,
                        "incompatibility": incompatibility,
                        "selection_index": selection_index,
                    }
                )

            if not candidates:
                break
            best = max(
                candidates,
                key=lambda item: (
                    item["selection_index"],
                    item["coverage_gain"],
                    item["channel_quality"],
                    -item["backhaul_hops"],
                    self._id_sort_key(item["device"]),
                ),
            )
            device_id = best["device"]
            selected.append(device_id)
            newly_owned = []
            for server_id in sorted(best["covered_by_server"]):
                for feature in sorted(best["covered_by_server"][server_id]):
                    demand = (server_id, feature)
                    if demand in ownership:
                        continue
                    ownership[demand] = device_id
                    remaining[server_id].discard(feature)
                    newly_owned.append([server_id, feature])
            coverage_iteration += 1
            report.append(
                {
                    "stage": "coverage",
                    "iteration": coverage_iteration,
                    "device": device_id,
                    "origin_candidate": best["origin"],
                    "coverage_gain": best["coverage_gain"],
                    "channel_quality": best["channel_quality"],
                    "singleton_bandwidth": best["singleton_bandwidth"],
                    "backhaul_hops": best["backhaul_hops"],
                    "incompatibility": self._updated_json_number(best["incompatibility"]),
                    "selection_index": best["selection_index"],
                    "newly_owned_demands": newly_owned,
                }
            )

        # Stage 2 is meaningful only after Stage 1 has established coverage.
        if not any(remaining.values()):
            self._updated_localize_ownership(selected, ownership, report)

        # Only IoTs that no longer own any destination demand are pruned.
        active_owners = set(ownership.values())
        pruned = [device_id for device_id in selected if device_id not in active_owners]
        if pruned:
            selected[:] = [device_id for device_id in selected if device_id in active_owners]
            report.append(
                {
                    "stage": "pruning",
                    "removed_devices": pruned,
                    "rule": "empty_destination_set_only",
                }
            )

        return selected, ownership, remaining, report

    def _updated_localize_ownership(
        self,
        selected: list[DeviceId],
        ownership: Dict[Demand, DeviceId],
        report: list[dict[str, Any]],
    ) -> None:
        iteration = 0
        epsilon = 1e-12
        while True:
            current_hops = self._updated_total_backhaul_hops(ownership)
            trials: list[dict[str, Any]] = []
            for device_id in self.devices:
                if device_id in selected:
                    continue
                origin = self._updated_best_origin(device_id)
                if origin is None:
                    continue
                local_demands = {
                    demand
                    for demand in ownership
                    if demand[0] == origin
                    and demand[1] in self.devices[device_id].features
                    and ownership[demand] != device_id
                }
                if not local_demands:
                    continue
                trial_ownership = dict(ownership)
                for demand in local_demands:
                    trial_ownership[demand] = device_id
                trial_hops = self._updated_total_backhaul_hops(trial_ownership)
                hop_reduction = max(0.0, current_hops - trial_hops)
                if hop_reduction <= epsilon:
                    continue
                singleton_bandwidth = self._updated_singleton_bandwidth(
                    device_id,
                    origin,
                    {feature for _, feature in local_demands},
                )
                incompatibility = self._cfrg_selected_incompatibility(device_id, selected)
                trials.append(
                    {
                        "device": device_id,
                        "origin": origin,
                        "local_demands": local_demands,
                        "trial_ownership": trial_ownership,
                        "hop_reduction": hop_reduction,
                        "trial_hops": trial_hops,
                        "singleton_bandwidth": singleton_bandwidth,
                        "bandwidth_cost": self.config.c_bw * singleton_bandwidth,
                        "incompatibility": incompatibility,
                    }
                )

            if not trials:
                break

            hop_scale = max(
                current_hops,
                max(item["hop_reduction"] for item in trials),
                epsilon,
            )
            finite_bandwidth_costs = [
                item["bandwidth_cost"]
                for item in trials
                if math.isfinite(item["bandwidth_cost"])
            ]
            bandwidth_scale = max(finite_bandwidth_costs, default=epsilon)
            finite_incompatibilities = [
                item["incompatibility"]
                for item in trials
                if math.isfinite(item["incompatibility"])
            ]
            incompatibility_scale = max(finite_incompatibilities, default=1.0)
            if incompatibility_scale <= epsilon:
                incompatibility_scale = 1.0

            for item in trials:
                normalized_hops = item["hop_reduction"] / hop_scale
                normalized_bandwidth = item["bandwidth_cost"] / max(
                    bandwidth_scale,
                    epsilon,
                )
                normalized_incompatibility = (
                    item["incompatibility"] / incompatibility_scale
                    if math.isfinite(item["incompatibility"])
                    else float("inf")
                )
                denominator = (
                    normalized_bandwidth * (1.0 + normalized_incompatibility)
                    + epsilon
                )
                item["normalized_hop_reduction"] = normalized_hops
                item["normalized_bandwidth_cost"] = normalized_bandwidth
                item["normalized_incompatibility"] = normalized_incompatibility
                item["localization_index"] = self._updated_safe_ratio(
                    normalized_hops,
                    denominator,
                )

            best = max(
                trials,
                key=lambda item: (
                    item["localization_index"],
                    item["hop_reduction"],
                    -item["bandwidth_cost"],
                    self._id_sort_key(item["device"]),
                ),
            )
            if best["localization_index"] <= 1.0:
                report.append(
                    {
                        "stage": "localization_stop",
                        "reason": "best_index_not_greater_than_one",
                        "best_device": best["device"],
                        "best_localization_index": best["localization_index"],
                    }
                )
                break

            ownership.clear()
            ownership.update(best["trial_ownership"])
            selected.append(best["device"])
            iteration += 1
            report.append(
                {
                    "stage": "localization",
                    "iteration": iteration,
                    "device": best["device"],
                    "origin_candidate": best["origin"],
                    "reassigned_demands": [
                        [server_id, feature]
                        for server_id, feature in sorted(best["local_demands"])
                    ],
                    "hop_reduction": best["hop_reduction"],
                    "hops_after": best["trial_hops"],
                    "singleton_bandwidth": best["singleton_bandwidth"],
                    "bandwidth_cost": best["bandwidth_cost"],
                    "incompatibility": self._updated_json_number(best["incompatibility"]),
                    "normalized_hop_reduction": best["normalized_hop_reduction"],
                    "normalized_bandwidth_cost": best["normalized_bandwidth_cost"],
                    "normalized_incompatibility": self._updated_json_number(
                        best["normalized_incompatibility"]
                    ),
                    "localization_index": best["localization_index"],
                }
            )

    def _cfrg_reachable_servers(self, device_id: DeviceId) -> Set[ServerId]:
        device = self.devices[device_id]
        reachable = {
            server_id
            for server_id in self.servers
            if self._device_can_transmit_to_server(device_id, server_id)
        }
        if not reachable and device.home_server in self.servers:
            reachable.add(device.home_server)
        return reachable

    def _cfrg_selected_incompatibility(
        self,
        device_id: DeviceId,
        selected: Sequence[DeviceId],
    ) -> float:
        if not selected:
            return 0.0
        values = [
            value
            for other_id in selected
            for value in [self._cfrg_pair_dissimilarity(device_id, other_id)[0]]
            if value is not None
        ]
        return min(values) if values else float("inf")

    def _updated_best_origin(self, device_id: DeviceId) -> ServerId | None:
        reachable = self._cfrg_reachable_servers(device_id)
        if not reachable:
            return None
        return max(
            reachable,
            key=lambda server_id: (
                self.evaluator.channel_gain(device_id, server_id) ** 2,
                server_id,
            ),
        )

    def _updated_singleton_bandwidth(
        self,
        device_id: DeviceId,
        origin: ServerId,
        required_features: Set[str],
    ) -> float:
        group = self._make_group(
            origin,
            "updated_selection_candidate",
            (device_id,),
            required_features,
        )
        try:
            return self.evaluator.derive_group_bandwidth_for_budget(
                group,
                self.evaluator.bandwidth_time_budget(),
            )
        except Exception:
            return float("inf")

    def _updated_total_backhaul_hops(
        self,
        ownership: Mapping[Demand, DeviceId],
    ) -> float:
        destinations: Dict[DeviceId, Set[ServerId]] = {}
        for (server_id, _), device_id in ownership.items():
            destinations.setdefault(device_id, set()).add(server_id)
        total = 0.0
        for device_id, servers in destinations.items():
            origin = self._updated_best_origin(device_id)
            if origin is None:
                return float("inf")
            total += sum(
                self.evaluator.wired_weight(origin, server_id)
                for server_id in servers
            )
        return total

    @staticmethod
    def _updated_safe_ratio(numerator: float, denominator: float) -> float:
        if numerator <= 0.0 or math.isnan(numerator) or math.isnan(denominator):
            return 0.0
        if denominator <= 0.0:
            return float("inf")
        if math.isinf(denominator):
            return 0.0
        return numerator / denominator

    @staticmethod
    def _updated_json_number(value: float) -> float | None:
        return value if math.isfinite(value) else None

    def _cfrg_pair_dissimilarity(
        self,
        first: DeviceId,
        second: DeviceId,
    ) -> tuple[float | None, ServerId | None]:
        common_servers = self._cfrg_reachable_servers(first) & self._cfrg_reachable_servers(second)
        if not common_servers:
            return None, None
        distance = math.dist(self.devices[first].position, self.devices[second].position)
        candidates = []
        for server_id in common_servers:
            first_phase = float(self.devices[first].phase_angle.get(server_id, 0.0))
            second_phase = float(self.devices[second].phase_angle.get(server_id, 0.0))
            value = distance * (2.0 - math.cos(first_phase - second_phase))
            candidates.append((value, server_id))
        return min(candidates, key=lambda item: (item[0], item[1]))

    def _cfrg_mst_order(
        self,
        selected: Sequence[DeviceId],
    ) -> tuple[list[DeviceId], dict[str, Any]]:
        devices = sorted(set(selected), key=self._id_sort_key)
        pair_values: Dict[Tuple[DeviceId, DeviceId], float] = {}
        pair_report: list[dict[str, Any]] = []
        for index, first in enumerate(devices):
            for second in devices[index + 1 :]:
                value, server_id = self._cfrg_pair_dissimilarity(first, second)
                if value is not None:
                    pair_values[(first, second)] = value
                pair_report.append(
                    {
                        "first": first,
                        "second": second,
                        "dissimilarity": value,
                        "minimizing_server": server_id,
                    }
                )

        tree: Dict[DeviceId, list[Tuple[float, DeviceId]]] = {
            device_id: [] for device_id in devices
        }
        unvisited = set(devices)
        roots: list[DeviceId] = []
        mst_edges: list[dict[str, Any]] = []
        while unvisited:
            root = min(unvisited, key=self._id_sort_key)
            roots.append(root)
            component = {root}
            unvisited.remove(root)
            while unvisited:
                frontier = []
                for source in component:
                    for target in unvisited:
                        pair = tuple(sorted((source, target), key=self._id_sort_key))
                        value = pair_values.get(pair)
                        if value is not None:
                            frontier.append((value, source, target))
                if not frontier:
                    break
                value, source, target = min(
                    frontier,
                    key=lambda item: (
                        item[0],
                        self._id_sort_key(item[1]),
                        self._id_sort_key(item[2]),
                    ),
                )
                component.add(target)
                unvisited.remove(target)
                tree[source].append((value, target))
                tree[target].append((value, source))
                mst_edges.append(
                    {"source": source, "target": target, "weight": value}
                )

        order: list[DeviceId] = []
        visited: Set[DeviceId] = set()

        for root in roots:
            if root in visited:
                continue
            queue = deque([root])
            visited.add(root)
            while queue:
                device_id = queue.popleft()
                order.append(device_id)
                for _, neighbor in sorted(
                    tree[device_id],
                    key=lambda item: (item[0], self._id_sort_key(item[1])),
                ):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order, {
            "roots": roots,
            "traversal": "BFS",
            "pairwise_compatibility": pair_report,
            "edges": mst_edges,
        }

    def _cfrg_interval_groups(
        self,
        order: Sequence[DeviceId],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> tuple[list[GroupSpec], list[dict[str, Any]], dict[str, Any]]:
        count = len(order)
        if count == 0:
            return [], [], {"objective": 0.0, "predecessors": []}
        group_limit = max(1, int(self.config.max_group_size))
        dp = [-float("inf")] * (count + 1)
        predecessor = [-1] * (count + 1)
        interval_cache: Dict[Tuple[int, int], dict[str, Any]] = {}
        dp[0] = 0.0

        for end in range(1, count + 1):
            for length in range(1, min(group_limit, end) + 1):
                start = end - length
                info = self._cfrg_interval_info(order[start:end], ownership)
                interval_cache[(start, end)] = info
                if not info["feasible"] or dp[start] == -float("inf"):
                    continue
                value = dp[start] + info["utility"]
                if value > dp[end] + 1e-12:
                    dp[end] = value
                    predecessor[end] = start

        if dp[count] == -float("inf"):
            return [], [], {
                "objective": None,
                "predecessors": predecessor,
                "infeasible": True,
            }

        intervals = []
        cursor = count
        while cursor > 0:
            start = predecessor[cursor]
            if start < 0:
                return [], [], {
                    "objective": None,
                    "predecessors": predecessor,
                    "infeasible": True,
                }
            intervals.append((start, cursor))
            cursor = start
        intervals.reverse()

        groups: list[GroupSpec] = []
        records: list[dict[str, Any]] = []
        for group_index, interval in enumerate(intervals):
            info = dict(interval_cache[interval])
            group = self._make_group(
                info["origin_server"],
                f"g{group_index}",
                tuple(info["members"]),
                set(info["required_features"]),
            )
            groups.append(group)
            info["group_id"] = group.id
            info["destinations"] = {
                server_id: sorted(features)
                for server_id, features in info["destinations"].items()
            }
            info["required_features"] = sorted(info["required_features"])
            records.append(info)

        return groups, records, {
            "objective": dp[count],
            "predecessors": predecessor,
            "intervals": [list(interval) for interval in intervals],
            "group_limit": group_limit,
        }

    def _cfrg_interval_info(
        self,
        members: Sequence[DeviceId],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> dict[str, Any]:
        delivery = self._cfrg_delivery_info(members, ownership)
        if delivery is None:
            return {
                "feasible": False,
                "members": list(members),
                "utility": -float("inf"),
            }
        singleton_bandwidth = 0.0
        singleton_details = []
        for device_id in members:
            singleton = self._cfrg_delivery_info([device_id], ownership)
            if singleton is None:
                return {
                    "feasible": False,
                    "members": list(members),
                    "utility": -float("inf"),
                }
            singleton_bandwidth += singleton["bandwidth"]
            singleton_details.append(
                {
                    "device": device_id,
                    "origin_server": singleton["origin_server"],
                    "bandwidth": singleton["bandwidth"],
                }
            )
        bandwidth_saving = singleton_bandwidth - delivery["bandwidth"]
        backhaul_weight = sum(
            self.evaluator.wired_weight(delivery["origin_server"], server_id)
            for server_id in delivery["destinations"]
        )
        forwarding_cost = self.config.c_fwd * backhaul_weight
        utility = self.config.c_bw * bandwidth_saving - forwarding_cost
        return {
            "feasible": True,
            **delivery,
            "singleton_details": singleton_details,
            "bandwidth_without_grouping": singleton_bandwidth,
            "bandwidth_saving": bandwidth_saving,
            "backhaul_weight": backhaul_weight,
            "forwarding_cost": forwarding_cost,
            "utility": utility,
        }

    def _cfrg_delivery_info(
        self,
        members: Sequence[DeviceId],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> dict[str, Any] | None:
        if not members:
            return None
        feasible_servers = set(self.servers)
        for device_id in members:
            feasible_servers &= self._cfrg_reachable_servers(device_id)
        if not feasible_servers:
            return None
        origin = max(
            feasible_servers,
            key=lambda server_id: (
                min(
                    self.evaluator.channel_gain(device_id, server_id) ** 2
                    for device_id in members
                ),
                server_id,
            ),
        )
        worst_channel_gain = min(
            self.evaluator.channel_gain(device_id, origin) ** 2
            for device_id in members
        )
        member_set = set(members)
        destinations: Dict[ServerId, Set[str]] = defaultdict(set)
        for (server_id, feature), owner in ownership.items():
            if owner in member_set:
                destinations[server_id].add(feature)
        required_features = set().union(*destinations.values()) if destinations else set()
        group = self._make_group(
            origin,
            "candidate",
            tuple(members),
            required_features,
        )
        dependencies: list[tuple[AssignmentKey, ServerId]] = []
        assignments = getattr(self, "_cfrg_assignments", {})
        latest_start = getattr(self, "_cfrg_latest_start", {})
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                assigned_servers = self.evaluator.participating_servers(assignments, key)
                for target_server in assigned_servers:
                    delivered = destinations.get(target_server, set())
                    if set(subtask.required_features) & delivered:
                        dependencies.append((key, target_server))

        if dependencies:
            budgets = []
            for key, target_server in dependencies:
                task = self.tasks[key[0]]
                if task.deadline == float("inf"):
                    remaining = self.evaluator.bandwidth_time_budget()
                else:
                    remaining = latest_start.get(key, task.deadline)
                if origin != target_server:
                    remaining -= (
                        self.evaluator.group_feature_volume(group)
                        / self.evaluator.wired_rate(origin, target_server)
                    )
                budgets.append(remaining)
            time_budget = min(budgets)
        else:
            time_budget = self.evaluator.bandwidth_time_budget()
        if time_budget <= 0.0:
            return None

        bandwidth = self.evaluator.derive_group_bandwidth_for_budget(
            group,
            time_budget,
        )
        return {
            "members": list(members),
            "feasible_servers": sorted(feasible_servers),
            "origin_server": origin,
            "worst_channel_gain": worst_channel_gain,
            "destinations": dict(destinations),
            "required_features": required_features,
            "dependent_subtasks": [
                {
                    "subtask": self._assignment_label(key),
                    "destination": target_server,
                }
                for key, target_server in dependencies
            ],
            "time_budget": time_budget,
            "bandwidth": bandwidth,
        }

    def _cfrg_backhaul_plan(
        self,
        group_records: Sequence[Mapping[str, Any]],
    ) -> tuple[
        Set[Tuple[ServerId, str, ServerId]],
        list[ChainedBackhaulDecision],
    ]:
        backhaul: Set[Tuple[ServerId, str, ServerId]] = set()
        plan: list[ChainedBackhaulDecision] = []
        for record in group_records:
            origin = str(record["origin_server"])
            group_id = str(record["group_id"])
            transmitted_features = sorted(
                {
                    feature
                    for device_id in record["members"]
                    for feature in self.devices[device_id].features
                }
            )
            for target_server, features in record["destinations"].items():
                if target_server == origin or not features:
                    continue
                backhaul.add((origin, group_id, target_server))
                plan.append(
                    ChainedBackhaulDecision(
                        source_server=origin,
                        group_id=group_id,
                        target_server=target_server,
                        features=transmitted_features,
                    )
                )
        plan.sort(key=lambda item: (item.source_server, item.group_id, item.target_server))
        return backhaul, plan

    def _cfrg_subtask_features(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, str]]],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> Dict[Tuple[str, str, str], Set[str]]:
        owned_by_server: Dict[ServerId, Set[str]] = defaultdict(set)
        for server_id, feature in ownership:
            owned_by_server[server_id].add(feature)
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        subtask_by_key = {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }
        for key, pairs in assignments.items():
            subtask = subtask_by_key.get(key)
            if subtask is None:
                continue
            for server_id in {server_id for server_id, _ in pairs}:
                subtask_features[self._server_feature_key(key, server_id)] = (
                    set(subtask.required_features) & owned_by_server.get(server_id, set())
                )
        return subtask_features
