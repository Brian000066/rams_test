from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import heapq
from itertools import product
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_location_aware_backhaul import (
    ChainedBackhaulDecision,
    ChainedPipelineResult,
)
from utils.formulation import (
    AssignmentKey,
    ExpertId,
    FormulationEvaluator,
    GroupSpec,
    InfeasibleBandwidthBudget,
    ServerId,
    SubtaskSpec,
)


class Phase3RefinementPruningMixin:
    """Phase III: FAR flow-aligned remapping and redundancy pruning."""

    def _phase3_refine(
        self,
        assignments,
        current_result: ChainedPipelineResult,
    ) -> ChainedPipelineResult:
        """Run FAR exactly after SCOPE and CFRG have produced a feasible state."""
        initial_cost = current_result.total_cost
        self._far_fixed_groups = tuple(
            replace(
                group,
                bandwidth=0.0,
                required_features=self.evaluator.group_transmitted_features(group),
            )
            for group in self._groups_from_result(current_result)
        )
        self.phase3_report["fixed_group_membership"] = [
            {
                "origin_server": group.server_id,
                "group_id": group.id,
                "members": list(group.devices),
            }
            for group in self._far_fixed_groups
        ]
        working = {
            key: list(value)
            for key, value in current_result.subtask_assignment.items()
        }
        best_result, working, pruning_report = self._far_prune_redundancy(
            working,
            current_result,
        )
        context = self._scope_context()
        structures = list(self.phase1_report.get("shared_structures", []))
        max_iterations = max(1, int(self.phase3_report.get("max_iterations", 2)))
        max_joint = max(1, int(self.phase3_report.get("max_joint_assignments", 32)))
        max_private = max(1, int(self.phase3_report.get("max_private_candidates", 48)))
        accepted_moves: list[dict[str, Any]] = []
        trace_report: list[dict[str, Any]] = []
        candidate_evaluations = 0

        for iteration in range(max_iterations):
            improved = False
            for structure in structures:
                paths = self._far_enumerate_paths(structure)
                placement = self._far_placement(working)
                blocks, diagnostics = self._far_marked_blocks(
                    structure,
                    paths,
                    placement,
                    context,
                )
                trace_report.append(
                    {
                        "iteration": iteration + 1,
                        "structure": structure.get("id"),
                        "paths": diagnostics,
                        "marked_blocks": blocks,
                    }
                )
                for block in blocks:
                    candidate_sets = self._far_shared_candidate_sets(
                        block,
                        paths,
                        placement,
                        structure,
                        context,
                    )
                    best_candidate = None
                    for choice in self._far_joint_choices(candidate_sets, max_joint):
                        trial_placement = self._far_apply_shared_move(
                            placement,
                            structure,
                            choice,
                            context,
                        )
                        if trial_placement == placement:
                            continue
                        trial_result = self._far_repair_and_evaluate(trial_placement)
                        candidate_evaluations += 1
                        if trial_result is None:
                            continue
                        if trial_result.total_cost >= best_result.total_cost:
                            continue
                        item = (trial_result.total_cost, choice, trial_result)
                        if best_candidate is None or item[0] < best_candidate[0]:
                            best_candidate = item
                    if best_candidate is None:
                        continue
                    _, choice, accepted_result = best_candidate
                    cost_before = best_result.total_cost
                    best_result = accepted_result
                    working = {
                        key: list(value)
                        for key, value in best_result.subtask_assignment.items()
                    }
                    accepted_moves.append(
                        {
                            "stage": "shared_structure",
                            "iteration": iteration + 1,
                            "structure": structure.get("id"),
                            "block": list(block),
                            "joint_assignment": dict(choice),
                            "cost_before": cost_before,
                            "cost_after": best_result.total_cost,
                            "saved_cost": cost_before - best_result.total_cost,
                        }
                    )
                    improved = True
            if not improved:
                break

        shared_keys = self._far_shared_occurrences(structures, context)
        private_report: list[dict[str, Any]] = []
        private_evaluations = 0
        for key in self._far_private_order(shared_keys):
            if private_evaluations >= max_private:
                break
            placement = self._far_placement(working)
            candidates, references = self._far_private_candidates(
                key,
                placement,
                shared_keys,
                context,
            )
            best_candidate = None
            evaluated_servers = []
            for server_id in candidates:
                if private_evaluations >= max_private:
                    break
                if server_id == placement.get(key):
                    continue
                trial_placement = dict(placement)
                trial_placement[key] = server_id
                trial_result = self._far_repair_and_evaluate(trial_placement)
                private_evaluations += 1
                candidate_evaluations += 1
                evaluated_servers.append(server_id)
                if trial_result is None:
                    continue
                if trial_result.total_cost >= best_result.total_cost:
                    continue
                item = (trial_result.total_cost, server_id, trial_result)
                if best_candidate is None or item[:2] < best_candidate[:2]:
                    best_candidate = item
            record = {
                "subtask": self._assignment_label(key),
                "references": [self._assignment_label(item) for item in references],
                "candidate_servers": candidates,
                "evaluated_servers": evaluated_servers,
                "accepted": best_candidate is not None,
            }
            if best_candidate is not None:
                _, selected_server, accepted_result = best_candidate
                cost_before = best_result.total_cost
                best_result = accepted_result
                working = {
                    item_key: list(value)
                    for item_key, value in best_result.subtask_assignment.items()
                }
                record.update(
                    {
                        "selected_server": selected_server,
                        "cost_before": cost_before,
                        "cost_after": best_result.total_cost,
                        "saved_cost": cost_before - best_result.total_cost,
                    }
                )
                accepted_moves.append({"stage": "private_propagation", **record})
            private_report.append(record)

        self.phase3_report.update(
            {
                "mode": "FAR",
                "redundancy_pruning": pruning_report,
                "shared_structure_traces": trace_report,
                "private_propagation": private_report,
                "candidate_evaluations": candidate_evaluations,
                "accepted_moves": accepted_moves,
                "cost_before": initial_cost,
                "cost_after": best_result.total_cost,
                "saved_cost": initial_cost - best_result.total_cost,
                "constraint_violations_after": len(best_result.violations),
            }
        )
        setattr(best_result, "phase3_report", self.phase3_report)
        return best_result

    def _far_placement(self, assignments) -> Dict[AssignmentKey, ServerId]:
        return {
            key: pairs[0][0]
            for key, pairs in assignments.items()
            if pairs
        }

    def _far_enumerate_paths(self, structure) -> list[list[str]]:
        nodes = list(structure.get("node_types", []))
        edges = [tuple(edge) for edge in structure.get("edges", [])]
        successors: Dict[str, list[str]] = {node: [] for node in nodes}
        indegree = {node: 0 for node in nodes}
        for source, target in edges:
            if source in successors and target in successors:
                successors[source].append(target)
                indegree[target] += 1
        sources = [node for node in nodes if indegree[node] == 0]
        sinks = {node for node in nodes if not successors[node]}
        paths: list[list[str]] = []

        def visit(node, path, seen):
            if len(paths) >= 128:
                return
            if node in sinks:
                paths.append(path + [node])
                return
            for target in sorted(successors[node]):
                if target not in seen:
                    visit(target, path + [node], seen | {target})

        for source in sorted(sources):
            visit(source, [], {source})
        return paths or [[node] for node in nodes]

    def _far_node_type_server(self, node_type, structure, placement, context):
        for task_id in structure.get("support_tasks", []):
            key = context["occurrence_by_task_type"].get((task_id, node_type))
            if key in placement:
                return placement[key]
        for key in context["occurrences_by_type"].get(node_type, []):
            if key in placement:
                return placement[key]
        return structure.get("placement_by_type", {}).get(node_type)

    def _far_marked_blocks(self, structure, paths, placement, context):
        blocks: list[list[str]] = []
        diagnostics = []
        for path in paths:
            servers = [
                self._far_node_type_server(node_type, structure, placement, context)
                for node_type in path
            ]
            if not servers or any(server_id is None for server_id in servers):
                continue
            trace = [servers[0]]
            spans = []
            for index in range(len(servers) - 1):
                physical = self._far_shortest_path(servers[index], servers[index + 1])
                start = len(trace) - 1
                trace.extend(physical[1:])
                spans.append((start, len(trace) - 1))
            marked, reasons = self._far_detect_detours(trace, spans)
            ordered = sorted(marked)
            path_blocks: list[list[str]] = []
            if ordered:
                current = [ordered[0]]
                for index in ordered[1:]:
                    if index == current[-1] + 1:
                        current.append(index)
                    else:
                        path_blocks.append([path[item] for item in current])
                        current = [index]
                path_blocks.append([path[item] for item in current])
            for block in path_blocks:
                if block not in blocks:
                    blocks.append(block)
            diagnostics.append(
                {
                    "node_path": path,
                    "node_servers": servers,
                    "physical_trace": trace,
                    "detours": reasons,
                    "marked_nodes": [path[index] for index in ordered],
                }
            )
        return blocks, diagnostics

    def _far_detect_detours(self, trace, spans):
        marked: Set[int] = set()
        reasons: list[dict[str, Any]] = []
        directed: Dict[Tuple[ServerId, ServerId], list[int]] = defaultdict(list)
        for index in range(len(trace) - 1):
            directed[(trace[index], trace[index + 1])].append(index)
        for (source, target), positions in directed.items():
            reverse = directed.get((target, source), [])
            if not reverse or str(source) > str(target):
                continue
            all_positions = positions + reverse
            self._far_mark_trace_range(marked, spans, min(all_positions), max(all_positions) + 1)
            reasons.append(
                {
                    "type": "reversed_link",
                    "link": [source, target],
                    "forward_positions": positions,
                    "reverse_positions": reverse,
                }
            )
        for left in range(len(trace)):
            for right in range(left + 2, len(trace)):
                segment = trace[left : right + 1]
                if trace[left] == trace[right] and segment == list(reversed(segment)):
                    self._far_mark_trace_range(marked, spans, left, right)
                    reasons.append(
                        {"type": "palindrome", "range": [left, right], "trace": segment}
                    )
        return marked, reasons

    @staticmethod
    def _far_mark_trace_range(marked, spans, left, right):
        for transition, (start, end) in enumerate(spans):
            if end >= left and start <= right:
                marked.add(transition)
                marked.add(transition + 1)

    def _far_physical_adjacency(self):
        adjacency: Dict[ServerId, list[Tuple[float, ServerId]]] = {
            server_id: [] for server_id in self.servers
        }
        for link in getattr(self, "physical_wired_links", ()):
            source = str(link.get("src"))
            target = str(link.get("dst"))
            weight = float(link.get("weight", 1.0))
            if source in adjacency and target in adjacency:
                adjacency[source].append((weight, target))
                adjacency[target].append((weight, source))
        return adjacency

    def _far_shortest_path(self, source, target):
        if source == target:
            return [source]
        adjacency = self._far_physical_adjacency()
        if not adjacency.get(source):
            return [source, target]
        queue = [(0.0, source, (source,))]
        best = {source: 0.0}
        while queue:
            cost, server_id, path = heapq.heappop(queue)
            if server_id == target:
                return list(path)
            if cost > best.get(server_id, float("inf")) + 1e-12:
                continue
            for weight, neighbor in adjacency.get(server_id, []):
                candidate = cost + weight
                if candidate < best.get(neighbor, float("inf")) - 1e-12:
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor, path + (neighbor,)))
        return [source, target]

    def _far_shared_candidate_sets(self, block, paths, placement, structure, context):
        candidate_sets = {}
        for node_type in block:
            candidates = {
                self._far_node_type_server(node_type, structure, placement, context)
            }
            for path in paths:
                for index, item in enumerate(path):
                    if item != node_type:
                        continue
                    for neighbor_index in (index - 1, index + 1):
                        if 0 <= neighbor_index < len(path):
                            server_id = self._far_node_type_server(
                                path[neighbor_index], structure, placement, context
                            )
                            if server_id is not None:
                                candidates.add(server_id)
            candidate_sets[node_type] = sorted(
                server_id for server_id in candidates if server_id in self.servers
            )
        return candidate_sets

    @staticmethod
    def _far_joint_choices(candidate_sets, limit):
        node_types = list(candidate_sets)
        for count, values in enumerate(
            product(*(candidate_sets[node_type] for node_type in node_types))
        ):
            if count >= limit:
                break
            yield dict(zip(node_types, values))

    def _far_apply_shared_move(self, placement, structure, choice, context):
        trial = dict(placement)
        for node_type, server_id in choice.items():
            for task_id in structure.get("support_tasks", []):
                key = context["occurrence_by_task_type"].get((task_id, node_type))
                if key is not None:
                    trial[key] = server_id
        return trial

    def _far_repair_and_evaluate(self, placement):
        """Repair X/Y by marginal loss reduction, then locally rerun CFRG."""
        subtask_by_key = self._subtask_by_key()
        keys_by_server: Dict[ServerId, list[AssignmentKey]] = defaultdict(list)
        for key, server_id in placement.items():
            if server_id not in self.servers or key not in subtask_by_key:
                return None
            keys_by_server[server_id].append(key)

        assignments: Dict[AssignmentKey, list[Tuple[ServerId, ExpertId]]] = {
            key: [] for key in placement
        }
        activated: Dict[ServerId, Set[ExpertId]] = {
            server_id: set() for server_id in self.servers
        }
        used_memory = {server_id: 0.0 for server_id in self.servers}
        probabilities = {key: 0.0 for key in placement}

        for server_id, keys in keys_by_server.items():
            server = self.servers[server_id]
            while True:
                unsatisfied = [
                    key
                    for key in keys
                    if self.evaluator.performance_loss_from_probability_and_reconstruction(
                        probabilities[key], 0.0
                    )
                    > self.evaluator.conformal_loss_threshold(subtask_by_key[key]) + 1e-12
                ]
                if not unsatisfied:
                    break
                candidates = []
                for expert_id in server.stored_experts:
                    if expert_id in activated[server_id] or expert_id not in self.experts:
                        continue
                    memory = self.experts[expert_id].memory
                    if used_memory[server_id] + memory > server.gpu_memory + 1e-12:
                        continue
                    reduction = 0.0
                    for key in unsatisfied:
                        contribution = self.evaluator.expert_contribution(
                            subtask_by_key[key], expert_id
                        )
                        old_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                            probabilities[key], 0.0
                        )
                        new_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                            probabilities[key] + contribution, 0.0
                        )
                        reduction += max(0.0, old_loss - new_loss)
                    candidates.append((reduction, -memory, expert_id))
                if not candidates:
                    return None
                reduction, _, expert_id = max(candidates)
                if reduction <= 1e-12:
                    return None
                activated[server_id].add(expert_id)
                used_memory[server_id] += self.experts[expert_id].memory
                for key in keys:
                    contribution = self.evaluator.expert_contribution(
                        subtask_by_key[key], expert_id
                    )
                    if contribution > 0.0:
                        assignments[key].append((server_id, expert_id))
                        probabilities[key] += contribution

        if any(not pairs for pairs in assignments.values()):
            return None
        selected_probability, _ = self._selection_metadata_from_assignments(assignments)
        result = self._far_evaluate_fixed_groups(assignments, selected_probability)
        return None if result is None or result.violations else result

    def _far_resolve_feature_sources(
        self,
        assignments,
        groups,
        preferred_source_map=None,
        base_backhaul=None,
        requested_subtask_features=None,
        *,
        allow_new_backhaul,
        preserve_base_backhaul,
    ):
        """Resolve every task-feature demand to one concrete RSMA group."""
        preferred = dict(preferred_source_map or {})
        base_edges = set(base_backhaul or set())
        resolved_edges = set(base_edges) if preserve_base_backhaul else set()
        group_by_key = {
            (group.server_id, group.id): group for group in groups
        }
        payload_by_key = {
            key: self.evaluator.group_transmitted_features(group)
            for key, group in group_by_key.items()
        }
        subtask_by_key = self._subtask_by_key()
        source_map = {}
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        missing = []

        def source_is_usable(source, target_server, feature):
            source = tuple(source)
            group = group_by_key.get(source)
            if group is None or feature not in payload_by_key[source]:
                return False
            if group.server_id == target_server:
                return True
            edge = (group.server_id, group.id, target_server)
            return edge in base_edges or allow_new_backhaul

        def source_order(source, target_server):
            group = group_by_key[source]
            if group.server_id == target_server:
                return (0, 0.0, str(group.server_id), str(group.id))
            edge = (group.server_id, group.id, target_server)
            route_rank = 1 if edge in base_edges else 2
            transfer_time = (
                self.evaluator.group_feature_volume(group)
                / self.evaluator.wired_rate(group.server_id, target_server)
            )
            return (
                route_rank,
                transfer_time,
                str(group.server_id),
                str(group.id),
            )

        for key in sorted(assignments):
            subtask = subtask_by_key.get(key)
            if subtask is None:
                continue
            for target_server in self.evaluator.participating_servers(
                assignments,
                key,
            ):
                feature_key = self._server_feature_key(key, target_server)
                received = subtask_features.setdefault(feature_key, set())
                requested_features = (
                    set(subtask.required_features)
                    if requested_subtask_features is None
                    else self.evaluator.features_for_server(
                        requested_subtask_features,
                        key,
                        target_server,
                    )
                )
                for feature in sorted(requested_features):
                    demand_key = (key[0], key[1], target_server, feature)
                    source = preferred.get(demand_key)
                    if source is not None and source_is_usable(
                        source,
                        target_server,
                        feature,
                    ):
                        source = tuple(source)
                    else:
                        candidates = [
                            source_key
                            for source_key, payload in payload_by_key.items()
                            if feature in payload
                            and source_is_usable(
                                source_key,
                                target_server,
                                feature,
                            )
                        ]
                        source = (
                            None
                            if not candidates
                            else min(
                                candidates,
                                key=lambda source_key: source_order(
                                    source_key,
                                    target_server,
                                ),
                            )
                        )
                    if source is None:
                        missing.append(
                            {
                                "subtask": self._assignment_label(key),
                                "server": target_server,
                                "feature": feature,
                            }
                        )
                        continue
                    source_map[demand_key] = source
                    received.add(feature)
                    if source[0] != target_server:
                        resolved_edges.add((source[0], source[1], target_server))

        plan = []
        for source_server, group_id, target_server in sorted(resolved_edges):
            group = group_by_key.get((source_server, group_id))
            if group is None:
                missing.append(
                    {
                        "source_server": source_server,
                        "group_id": group_id,
                        "server": target_server,
                        "feature": None,
                    }
                )
                continue
            plan.append(
                ChainedBackhaulDecision(
                    source_server=source_server,
                    group_id=group_id,
                    target_server=target_server,
                    features=sorted(
                        self.evaluator.group_transmitted_features(group)
                    ),
                )
            )
        return source_map, resolved_edges, plan, subtask_features, missing

    def _far_evaluate_fixed_groups(
        self,
        assignments,
        selected_probability,
        source_result=None,
    ):
        """Evaluate FAR placement while preserving Phase-II IoT group membership."""
        self._far_last_fixed_group_rejection = None
        groups = [
            replace(group, bandwidth=0.0)
            for group in getattr(self, "_far_fixed_groups", ())
        ]
        if not groups:
            self._far_last_fixed_group_rejection = {
                "reason": "no fixed Phase-II groups are available",
                "constraint_violations": [],
            }
            return None

        preferred_source_map = (
            None
            if source_result is None
            else source_result.feature_source_map
        )
        base_backhaul = (
            set()
            if source_result is None
            else set(source_result.backhaul)
        )
        (
            feature_source_map,
            backhaul,
            backhaul_plan,
            subtask_features,
            missing,
        ) = self._far_resolve_feature_sources(
            assignments,
            groups,
            preferred_source_map=preferred_source_map,
            base_backhaul=base_backhaul,
            allow_new_backhaul=True,
            preserve_base_backhaul=source_result is not None,
        )
        if missing:
            self._far_last_fixed_group_rejection = {
                "reason": "required feature would be unavailable",
                "missing_required_features": missing,
                "constraint_violations": [],
            }
            return None

        data_req = self._subtask_data_requirements(assignments, subtask_features)
        feature_to_groups = self._feature_to_groups(groups)
        try:
            resolved_groups, budgets = self._derive_group_bandwidths(
                groups,
                assignments,
                data_req,
                backhaul,
                feature_to_groups,
                feature_source_map=feature_source_map,
                subtask_features=subtask_features,
            )
        except InfeasibleBandwidthBudget as exc:
            self._far_last_fixed_group_rejection = {
                "reason": "non-positive uplink time budget",
                "detail": str(exc),
                "constraint_violations": [],
            }
            return None

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
            feature_source_map=feature_source_map,
        )
        if evaluation.violations:
            self._far_last_fixed_group_rejection = {
                "reason": "full formulation constraint check failed",
                "constraint_violations": list(evaluation.violations),
            }
            return None

        (
            _,
            _,
            required_probability,
            reconstruction_loss,
            performance_loss,
        ) = self._recompute_loss_metadata(
            assignments,
            selected_probability,
            resolved_groups,
            subtask_features,
        )
        server_features = self._server_required_features(data_req)

        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment={key: list(value) for key, value in assignments.items()},
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
            selected_probability=dict(selected_probability),
            required_probability=required_probability,
            reconstruction_loss=reconstruction_loss,
            performance_loss=performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=list(evaluation.violations),
            evaluation=evaluation,
            feature_source_map=feature_source_map,
        )

    def _far_shared_occurrences(self, structures, context):
        shared = set()
        for structure in structures:
            for task_id in structure.get("support_tasks", []):
                for node_type in structure.get("node_types", []):
                    key = context["occurrence_by_task_type"].get((task_id, node_type))
                    if key is not None:
                        shared.add(key)
        return shared

    def _far_private_order(self, shared_keys):
        ordered = []
        for task in sorted(self.tasks.values(), key=lambda item: item.id):
            remaining = {subtask.id: subtask for subtask in task.subtasks}
            emitted = set()
            while remaining:
                ready = [
                    subtask
                    for subtask in remaining.values()
                    if all(
                        predecessor in emitted or predecessor not in remaining
                        for predecessor in subtask.predecessors
                    )
                ]
                if not ready:
                    ready = list(remaining.values())
                for subtask in sorted(ready, key=lambda item: item.id):
                    key = (task.id, subtask.id)
                    if key not in shared_keys:
                        ordered.append(key)
                    emitted.add(subtask.id)
                    remaining.pop(subtask.id, None)
        return ordered

    def _far_private_candidates(self, key, placement, shared_keys, context):
        references = [
            neighbor
            for neighbor in context["neighbors"].get(key, set())
            if neighbor in shared_keys and neighbor in placement
        ]
        if not references:
            subtask = context["subtask_by_key"][key]
            references = [
                (key[0], predecessor)
                for predecessor in subtask.predecessors
                if (key[0], predecessor) in placement
            ]
        candidates = {placement[key]}
        adjacency = self._far_physical_adjacency()
        for reference in references:
            server_id = placement[reference]
            candidates.add(server_id)
            candidates.update(neighbor for _, neighbor in adjacency.get(server_id, []))
        return sorted(candidates), references

    def _far_server_task_node_counts(self, assignments):
        """Count distinct assigned task nodes served by each server."""
        counts: Dict[ServerId, int] = {
            server_id: 0 for server_id in self.servers
        }
        for pairs in assignments.values():
            for server_id in {pair[0] for pair in pairs}:
                counts[server_id] = counts.get(server_id, 0) + 1
        return counts

    @staticmethod
    def _far_server_priority_ranks(server_ids, task_node_counts):
        ordered = sorted(
            set(server_ids),
            key=lambda server_id: (
                -task_node_counts.get(server_id, 0),
                str(server_id),
            ),
        )
        return {
            server_id: rank
            for rank, server_id in enumerate(ordered, start=1)
        }

    def _far_phase1_expert_candidates(self, assignments, task_node_counts):
        """Return only Phase-I activated experts that remain in the state."""
        activation_report = self.phase1_report.get("expert_activation", {})
        existing_pairs = {
            pair for pairs in assignments.values() for pair in pairs
        }
        server_ids = {server_id for server_id, _ in existing_pairs}
        priority_ranks = self._far_server_priority_ranks(
            server_ids,
            task_node_counts,
        )
        candidates = []
        for server_id in sorted(
            server_ids,
            key=lambda item: (priority_ranks[item], str(item)),
        ):
            server_report = activation_report.get(
                server_id,
                activation_report.get(str(server_id), {}),
            )
            activated = set(server_report.get("activated_experts", ()))
            scores = server_report.get("activation_scores", {})
            for pair in sorted(
                (
                    pair
                    for pair in existing_pairs
                    if pair[0] == server_id and pair[1] in activated
                ),
                key=lambda item: str(item[1]),
            ):
                raw_score = scores.get(pair[1], scores.get(str(pair[1])))
                candidates.append(
                    {
                        "pair": pair,
                        "server": server_id,
                        "expert": pair[1],
                        "phase1_activation_score": raw_score,
                        "activation_score_order": (
                            float(raw_score)
                            if raw_score is not None
                            else float("inf")
                        ),
                        "task_node_count": task_node_counts.get(server_id, 0),
                        "server_priority_rank": priority_ranks[server_id],
                    }
                )
        return candidates

    def _far_prune_redundancy(self, assignments, current_result):
        """Prune Phase-I experts first, then Phase-II backhaul tuples.

        Both passes start from the server serving the most distinct task nodes.
        Every tentative removal is fully reevaluated and is accepted only when
        every constraint remains feasible and total cost strictly decreases.
        """
        result = current_result
        working = {key: list(value) for key, value in assignments.items()}
        removed_experts: list[dict[str, Any]] = []
        removed_backhaul: list[dict[str, Any]] = []
        accepted_removals: list[dict[str, Any]] = []
        evaluation_trace: list[dict[str, Any]] = []
        expert_round = 0
        while True:
            expert_round += 1
            task_node_counts = self._far_server_task_node_counts(working)
            expert_candidates = self._far_phase1_expert_candidates(
                working,
                task_node_counts,
            )
            accepted_candidate = None

            for node_count in sorted(
                {item["task_node_count"] for item in expert_candidates},
                reverse=True,
            ):
                priority_bucket = [
                    item
                    for item in expert_candidates
                    if item["task_node_count"] == node_count
                ]
                priority_bucket.sort(
                    key=lambda item: (
                        item["activation_score_order"],
                        item["server_priority_rank"],
                        str(item["expert"]),
                    )
                )
                for item in priority_bucket:
                    pair = item["pair"]
                    affected = [
                        key for key, pairs in working.items() if pair in pairs
                    ]
                    trial = {
                        key: [value for value in pairs if value != pair]
                        for key, pairs in working.items()
                    }
                    trace = {
                        "stage": "expert_pruning",
                        "round": expert_round,
                        "type": "expert",
                        "candidate_source": (
                            "phase1.expert_activation.activated_experts"
                        ),
                        "server": item["server"],
                        "expert": item["expert"],
                        "server_task_node_counts": dict(task_node_counts),
                        "server_task_node_count": item["task_node_count"],
                        "server_priority_rank": item["server_priority_rank"],
                        "phase1_activation_score": item[
                            "phase1_activation_score"
                        ],
                        "affected_subtasks": [
                            self._assignment_label(key) for key in affected
                        ],
                        "cost_before": result.total_cost,
                        "cost_after": None,
                        "constraint_violations": [],
                        "feasible": False,
                        "improving": False,
                        "accepted": False,
                    }
                    if any(not trial[key] for key in affected):
                        trace["rejection_reason"] = (
                            "an affected subtask would have no selected expert"
                        )
                        evaluation_trace.append(trace)
                        continue
                    if not self._far_full_feature_loss_feasible(trial, affected):
                        trace["rejection_reason"] = (
                            "performance-loss precheck failed"
                        )
                        evaluation_trace.append(trace)
                        continue

                    selected_probability, _ = (
                        self._selection_metadata_from_assignments(trial)
                    )
                    trial_result = self._far_evaluate_fixed_groups(
                        trial,
                        selected_probability,
                        source_result=result,
                    )
                    rejection = getattr(
                        self,
                        "_far_last_fixed_group_rejection",
                        None,
                    )
                    if trial_result is None:
                        if rejection:
                            trace["constraint_violations"] = list(
                                rejection.get("constraint_violations", [])
                            )
                            trace["rejection_reason"] = rejection.get(
                                "reason",
                                "full formulation evaluation failed",
                            )
                            for key, value in rejection.items():
                                if key not in {"reason", "constraint_violations"}:
                                    trace[key] = value
                        else:
                            trace["rejection_reason"] = (
                                "full formulation evaluation failed"
                            )
                        evaluation_trace.append(trace)
                        continue

                    trace["cost_after"] = trial_result.total_cost
                    trace["constraint_violations"] = list(
                        trial_result.violations
                    )
                    trace["feasible"] = not trial_result.violations
                    trace["improving"] = (
                        trace["feasible"]
                        and trial_result.total_cost < result.total_cost
                    )
                    if not trace["feasible"]:
                        trace["rejection_reason"] = "constraint violations"
                    elif not trace["improving"]:
                        trace["rejection_reason"] = (
                            "total cost did not strictly decrease"
                        )
                    evaluation_trace.append(trace)
                    if trace["improving"]:
                        accepted_candidate = (trial_result, trial, trace)
                        break

                if accepted_candidate is not None:
                    break

            if accepted_candidate is None:
                break

            accepted_result = accepted_candidate[0]
            accepted_assignments = accepted_candidate[1]
            accepted_trace = accepted_candidate[2]
            accepted_trace["accepted"] = True
            accepted_trace.pop("rejection_reason", None)
            accepted_trace["saved_cost"] = (
                result.total_cost - accepted_result.total_cost
            )
            accepted_removals.append(dict(accepted_trace))
            removed_experts.append(dict(accepted_trace))
            result = accepted_result
            working = accepted_assignments

        backhaul_round = 0
        while True:
            backhaul_round += 1
            task_node_counts = self._far_server_task_node_counts(working)
            edges = sorted(result.backhaul)
            destination_servers = {edge[2] for edge in edges}
            priority_ranks = self._far_server_priority_ranks(
                destination_servers,
                task_node_counts,
            )
            forwarding_cost_by_edge = {
                edge: self.config.c_fwd
                * self.evaluator.wired_weight(edge[0], edge[2])
                for edge in edges
            }
            accepted_candidate = None

            for node_count in sorted(
                {
                    task_node_counts.get(destination, 0)
                    for destination in destination_servers
                },
                reverse=True,
            ):
                priority_bucket = [
                    edge
                    for edge in edges
                    if task_node_counts.get(edge[2], 0) == node_count
                ]
                priority_bucket.sort(
                    key=lambda edge: (
                        -forwarding_cost_by_edge[edge],
                        priority_ranks[edge[2]],
                        str(edge[0]),
                        str(edge[1]),
                        str(edge[2]),
                    )
                )
                for edge in priority_bucket:
                    trial_result = self._far_remove_backhaul_tuple(result, edge)
                    cost_after = (
                        None if trial_result is None else trial_result.total_cost
                    )
                    violations = (
                        [] if trial_result is None else list(trial_result.violations)
                    )
                    feasible = trial_result is not None and not violations
                    improving = (
                        feasible
                        and cost_after is not None
                        and cost_after < result.total_cost
                    )
                    trace = {
                        "stage": "backhaul_pruning",
                        "round": backhaul_round,
                        "type": "backhaul",
                        "candidate_source": "phase2.backhaul",
                        "origin": edge[0],
                        "group": edge[1],
                        "destination": edge[2],
                        "server_task_node_counts": dict(task_node_counts),
                        "server_task_node_count": task_node_counts.get(
                            edge[2], 0
                        ),
                        "server_priority_rank": priority_ranks[edge[2]],
                        "forwarding_cost": forwarding_cost_by_edge[edge],
                        "cost_before": result.total_cost,
                        "cost_after": cost_after,
                        "constraint_violations": violations,
                        "feasible": feasible,
                        "improving": improving,
                        "accepted": False,
                        "feature_source_updates": list(
                            getattr(
                                self,
                                "_far_last_feature_source_updates",
                                [],
                            )
                        ),
                    }
                    if trial_result is None:
                        rejection = getattr(
                            self,
                            "_far_last_backhaul_rejection",
                            None,
                        )
                        if rejection:
                            trace.update(rejection)
                            trace["rejection_reason"] = rejection.get(
                                "reason",
                                "backhaul evaluation failed",
                            )
                        else:
                            trace["rejection_reason"] = (
                                "backhaul evaluation failed"
                            )
                    elif violations:
                        trace["rejection_reason"] = "constraint violations"
                    elif not improving:
                        trace["rejection_reason"] = (
                            "total cost did not strictly decrease"
                        )
                    evaluation_trace.append(trace)
                    if improving:
                        accepted_candidate = (trial_result, trace)
                        break

                if accepted_candidate is not None:
                    break

            if accepted_candidate is None:
                break

            accepted_result = accepted_candidate[0]
            accepted_trace = accepted_candidate[1]
            accepted_trace["accepted"] = True
            accepted_trace.pop("rejection_reason", None)
            accepted_trace["saved_cost"] = (
                result.total_cost - accepted_result.total_cost
            )
            accepted_removals.append(dict(accepted_trace))
            removed_backhaul.append(dict(accepted_trace))
            result = accepted_result
            working = {
                key: list(value)
                for key, value in result.subtask_assignment.items()
            }

        return result, working, {
            "acceptance_rule": (
                "tentative removal -> update affected variables -> full feasibility "
                "check -> accept iff total cost strictly decreases"
            ),
            "expert_candidate_source": (
                "Phase-I expert_activation.activated_experts intersected with "
                "the current assignments"
            ),
            "expert_priority_rule": (
                "descending server task-node count; ascending Phase-I "
                "activation score within the same count"
            ),
            "backhaul_candidate_source": "current Phase-II backhaul tuples",
            "backhaul_priority_rule": (
                "descending destination-server task-node count; descending "
                "forwarding cost within the same count"
            ),
            "removed_experts": removed_experts,
            "removed_backhaul_tuples": removed_backhaul,
            "accepted_removals": accepted_removals,
            "candidate_evaluations": evaluation_trace,
        }

    def _far_full_feature_loss_feasible(self, assignments, keys) -> bool:
        subtasks = self._subtask_by_key()
        for key in keys:
            subtask = subtasks.get(key)
            if subtask is None:
                continue
            experts = {expert_id for _, expert_id in assignments.get(key, [])}
            probability = self.evaluator.selection_probability(subtask, experts)
            loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                probability, 0.0
            )
            if loss > self.evaluator.conformal_loss_threshold(subtask) + 1e-12:
                return False
        return True

    def _far_remove_backhaul_tuple(self, result, edge):
        self._far_last_backhaul_rejection = None
        self._far_last_feature_source_updates = []
        groups = self._groups_from_result(result)
        if not groups:
            return None
        remaining_backhaul = set(result.backhaul) - {edge}
        (
            feature_source_map,
            resolved_backhaul,
            remaining_plan,
            subtask_features,
            missing_required_features,
        ) = self._far_resolve_feature_sources(
            result.subtask_assignment,
            groups,
            preferred_source_map=result.feature_source_map,
            base_backhaul=remaining_backhaul,
            allow_new_backhaul=False,
            preserve_base_backhaul=True,
        )
        if missing_required_features:
            self._far_last_backhaul_rejection = {
                "reason": "required feature would become unreachable",
                "missing_required_features": missing_required_features,
            }
            return None

        def serialized_source(source):
            if source is None:
                return None
            return {
                "source_server": source[0],
                "group_id": source[1],
            }

        changed_sources = []
        source_keys = set(result.feature_source_map) | set(feature_source_map)
        for demand_key in sorted(source_keys):
            old_source = result.feature_source_map.get(demand_key)
            new_source = feature_source_map.get(demand_key)
            if old_source == new_source:
                continue
            changed_sources.append(
                {
                    "demand": (
                        f"{demand_key[0]}:{demand_key[1]}:"
                        f"{demand_key[2]}:{demand_key[3]}"
                    ),
                    "before": serialized_source(old_source),
                    "after": serialized_source(new_source),
                }
            )
        self._far_last_feature_source_updates = changed_sources

        data_req = self._subtask_data_requirements(
            result.subtask_assignment, subtask_features
        )
        feature_to_groups = self._feature_to_groups(groups)
        try:
            resolved_groups, budgets = self._derive_group_bandwidths(
                groups,
                result.subtask_assignment,
                data_req,
                resolved_backhaul,
                feature_to_groups,
                feature_source_map=feature_source_map,
                subtask_features=subtask_features,
            )
        except InfeasibleBandwidthBudget as exc:
            self._far_last_backhaul_rejection = {
                "reason": "non-positive uplink time budget",
                "detail": str(exc),
            }
            return None
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(
            self.servers,
            self.devices,
            self.experts,
            self.tasks,
            eval_config,
        )
        evaluation = evaluator.evaluate(
            result.subtask_assignment,
            resolved_groups,
            resolved_backhaul,
            subtask_features=subtask_features,
            feature_source_map=feature_source_map,
        )
        _, _, required_probability, reconstruction_loss, performance_loss = (
            self._recompute_loss_metadata(
                result.subtask_assignment,
                result.selected_probability,
                resolved_groups,
                subtask_features,
            )
        )
        server_features = self._server_required_features(data_req)
        return ChainedPipelineResult(
            expert_placement=result.expert_placement,
            subtask_assignment=result.subtask_assignment,
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
                self._group_key_label(key): value for key, value in budgets.items()
            },
            rsma_group_bandwidths=self._group_bandwidth_map(resolved_groups),
            backhaul_plan=remaining_plan,
            selected_probability=result.selected_probability,
            required_probability=required_probability,
            reconstruction_loss=reconstruction_loss,
            performance_loss=performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=resolved_backhaul,
            violations=list(evaluation.violations),
            evaluation=evaluation,
            feature_source_map=feature_source_map,
        )

    def _legacy_phase3_refine(
        self,
        assignments,
        current_result: ChainedPipelineResult,
    ) -> ChainedPipelineResult:
        if not getattr(self, "_far_fixed_groups", None):
            self._far_fixed_groups = tuple(
                replace(
                    group,
                    bandwidth=0.0,
                    required_features=self.evaluator.group_transmitted_features(group),
                )
                for group in self._groups_from_result(current_result)
            )
        best_result = current_result
        working_assignments = {key: list(value) for key, value in assignments.items()}
        accepted_moves: list[Dict[str, Any]] = []
        candidate_evaluations = 0
        max_iterations = int(self.phase3_report.get("max_iterations", 3))
        max_candidates = int(self.phase3_report.get("max_candidates_per_strategy", 20))
        bandwidth_tolerance = float(self.phase3_report.get("bandwidth_increase_tolerance", 0.0))
        strategies = ("relocate_same_expert",)

        for iteration in range(max_iterations):
            accepted_this_iteration = False
            for strategy in strategies:
                best_candidate = None
                for move in self._phase3_candidate_moves(working_assignments, strategy, max_candidates):
                    trial_assignments = move.pop("assignments")
                    selected_probability, _ = self._selection_metadata_from_assignments(trial_assignments)
                    trial_result = self._far_evaluate_fixed_groups(
                        trial_assignments,
                        selected_probability,
                        source_result=best_result,
                    )
                    candidate_evaluations += 1
                    if trial_result is None or trial_result.violations:
                        continue
                    bandwidth_limit = best_result.bandwidth_cost * (1.0 + bandwidth_tolerance)
                    if trial_result.bandwidth_cost > bandwidth_limit + 1e-9:
                        continue
                    if trial_result.total_cost >= best_result.total_cost:
                        continue
                    candidate = (trial_result.total_cost, move, trial_result)
                    if best_candidate is None or candidate[0] < best_candidate[0]:
                        best_candidate = candidate

                if best_candidate is None:
                    continue

                _, accepted_move, accepted_result = best_candidate
                accepted_move = dict(accepted_move)
                accepted_move["iteration"] = iteration + 1
                accepted_move["cost_before"] = round(best_result.total_cost, 6)
                accepted_move["cost_after"] = round(accepted_result.total_cost, 6)
                accepted_move["saved_cost"] = round(best_result.total_cost - accepted_result.total_cost, 6)
                accepted_moves.append(accepted_move)
                best_result = accepted_result
                working_assignments = {
                    key: list(value) for key, value in accepted_result.subtask_assignment.items()
                }
                accepted_this_iteration = True
                break

            if not accepted_this_iteration:
                break

        self.phase3_report.update(
            {
                "iterations": len(accepted_moves),
                "candidate_evaluations": candidate_evaluations,
                "accepted_moves": accepted_moves,
                "cost_before": round(current_result.total_cost, 6),
                "cost_after": round(best_result.total_cost, 6),
                "saved_cost": round(current_result.total_cost - best_result.total_cost, 6),
            }
        )
        setattr(best_result, "phase3_report", self.phase3_report)
        return best_result

    def _run_offline_group_pruning(self, result: ChainedPipelineResult) -> ChainedPipelineResult:
        groups = self._groups_from_result(result)
        subtask_features = self._subtask_features_from_result(result)
        if not groups or not subtask_features:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "removed_groups": 0,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "No groups or subtask feature map available for pruning.",
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result

        pruned_groups, pruned_features = self._post_prune_groups(
            result.subtask_assignment,
            result.selected_probability,
            groups,
            subtask_features,
        )
        (
            feature_source_map,
            backhaul,
            plan,
            pruned_features,
            missing,
        ) = self._far_resolve_feature_sources(
            result.subtask_assignment,
            pruned_groups,
            preferred_source_map=result.feature_source_map,
            base_backhaul=result.backhaul,
            requested_subtask_features=pruned_features,
            allow_new_backhaul=True,
            preserve_base_backhaul=False,
        )
        if missing:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "reverted": True,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "Pruned candidate has unresolved feature sources.",
                    "missing_required_features": missing,
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result
        data_req = self._subtask_data_requirements(result.subtask_assignment, pruned_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(pruned_groups)
        try:
            resolved_groups, budgets = self._derive_group_bandwidths(
                pruned_groups,
                result.subtask_assignment,
                data_req,
                backhaul,
                feature_to_groups,
                feature_source_map=feature_source_map,
                subtask_features=pruned_features,
            )
        except InfeasibleBandwidthBudget:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "reverted": True,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "Pruned candidate has no positive uplink time budget.",
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        evaluation = evaluator.evaluate(
            result.subtask_assignment,
            resolved_groups,
            backhaul,
            subtask_features=pruned_features,
            feature_source_map=feature_source_map,
        )
        if evaluation.violations or evaluation.objective.total_cost > result.total_cost + 1e-9:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "reverted": True,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "Pruned candidate was not feasible or did not improve cost.",
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result

        pruned_result = ChainedPipelineResult(
            expert_placement=result.expert_placement,
            subtask_assignment=result.subtask_assignment,
            subtask_data_requirements=data_req,
            server_required_features={sid: sorted(value) for sid, value in server_features.items()},
            rsma_groups=self._group_map(resolved_groups),
            group_required_features={self._group_label(group): sorted(group.required_features) for group in resolved_groups},
            group_time_budgets={self._group_key_label(key): value for key, value in budgets.items()},
            rsma_group_bandwidths=self._group_bandwidth_map(resolved_groups),
            backhaul_plan=plan,
            selected_probability=result.selected_probability,
            required_probability=result.required_probability,
            reconstruction_loss=result.reconstruction_loss,
            performance_loss=result.performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=list(evaluation.violations),
            evaluation=evaluation,
            feature_source_map=feature_source_map,
        )
        setattr(pruned_result, "phase3_report", self.phase3_report)
        setattr(pruned_result, "post_pruning_report", self.post_pruning_report)
        return pruned_result

    def _groups_from_result(self, result: ChainedPipelineResult) -> list[GroupSpec]:
        group_features_by_server: Dict[ServerId, list[tuple[str, Set[str]]]] = {}
        for label, features in result.group_required_features.items():
            if ":" not in label:
                continue
            server_id, group_id = label.split(":", 1)
            group_features_by_server.setdefault(server_id, []).append((group_id, set(features)))

        fallback_required = set()
        for features in result.server_required_features.values():
            fallback_required.update(features)

        groups: list[GroupSpec] = []
        for server_id, member_groups in result.rsma_groups.items():
            feature_items = group_features_by_server.get(server_id, [])
            for index, members in enumerate(member_groups):
                if index < len(feature_items):
                    group_id, features = feature_items[index]
                else:
                    group_id = f"g{len(groups)}"
                    features = self._group_local_features(members, fallback_required)
                if not features:
                    continue
                groups.append(
                    self._make_group(
                        server_id=server_id,
                        group_id=group_id,
                        members=tuple(members),
                        required_features=features,
                    )
                )
        return groups

    def _subtask_features_from_result(self, result: ChainedPipelineResult):
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        for label, server_features in result.subtask_data_requirements.items():
            if ":" not in label:
                continue
            task_id, subtask_id = label.split(":", 1)
            for server_id, features in server_features.items():
                subtask_features[(task_id, subtask_id, server_id)] = set(features)
        return subtask_features

    def _phase3_candidate_moves(self, assignments, strategy: str, max_candidates: int):
        if strategy == "relocate_same_expert":
            yield from self._relocate_same_expert_moves(assignments, max_candidates)
            return
        if strategy == "remove_pair":
            count = 0
            for key in sorted(assignments):
                pairs = list(assignments[key])
                if len(pairs) <= 1:
                    continue
                for pair in pairs:
                    trial = {item_key: list(value) for item_key, value in assignments.items()}
                    trial[key] = [item for item in pairs if item != pair]
                    if not trial[key]:
                        continue
                    count += 1
                    yield {
                        "strategy": strategy,
                        "subtask": self._assignment_label(key),
                        "removed_pair": list(pair),
                        "assignments": trial,
                    }
                    if count >= max_candidates:
                        return
            return

        if strategy == "close_expert":
            active_pairs = sorted({pair for pairs in assignments.values() for pair in pairs})
            for pair in active_pairs[:max_candidates]:
                trial = {}
                feasible = True
                affected = 0
                for key, pairs in assignments.items():
                    next_pairs = [item for item in pairs if item != pair]
                    if len(next_pairs) != len(pairs):
                        affected += 1
                    if not next_pairs:
                        feasible = False
                        break
                    trial[key] = next_pairs
                if not feasible or affected <= 0:
                    continue
                yield {
                    "strategy": strategy,
                    "closed_pair": list(pair),
                    "affected_subtasks": affected,
                    "assignments": trial,
                }
            return

        if strategy == "replace_pair":
            count = 0
            subtask_by_key = self._subtask_by_key()
            for key in sorted(assignments):
                subtask = subtask_by_key.get(key)
                if subtask is None:
                    continue
                pairs = list(assignments[key])
                selected_experts = {expert_id for _, expert_id in pairs}
                ranked_experts = self._ranked_experts(subtask)[: max(self.top_k * 3, self.top_k + 4)]
                for old_pair in pairs:
                    for expert_id in ranked_experts:
                        if expert_id in selected_experts and expert_id != old_pair[1]:
                            continue
                        for server_id in self._feasible_servers_for_replacement(expert_id):
                            new_pair = (server_id, expert_id)
                            if new_pair == old_pair:
                                continue
                            if expert_id != old_pair[1] and any(existing_expert == expert_id for _, existing_expert in pairs):
                                continue
                            trial_pairs = [new_pair if item == old_pair else item for item in pairs]
                            if len({expert for _, expert in trial_pairs}) != len(trial_pairs):
                                continue
                            trial = {item_key: list(value) for item_key, value in assignments.items()}
                            trial[key] = trial_pairs
                            count += 1
                            yield {
                                "strategy": strategy,
                                "subtask": self._assignment_label(key),
                                "old_pair": list(old_pair),
                                "new_pair": list(new_pair),
                                "assignments": trial,
                            }
                            if count >= max_candidates:
                                return

    def _relocate_same_expert_moves(self, assignments, max_candidates: int):
        candidates = []
        for key in sorted(assignments):
            pairs = list(assignments[key])
            if len(pairs) <= 1:
                continue
            current_servers = {server_id for server_id, _ in pairs}
            for old_pair in pairs:
                old_server, expert_id = old_pair
                for target_server in sorted(current_servers):
                    if target_server == old_server:
                        continue
                    if expert_id not in self.servers[target_server].stored_experts:
                        continue
                    new_pair = (target_server, expert_id)
                    if new_pair in pairs:
                        continue
                    trial_pairs = [new_pair if item == old_pair else item for item in pairs]
                    if len(set(trial_pairs)) != len(trial_pairs):
                        continue
                    trial = {item_key: list(value) for item_key, value in assignments.items()}
                    trial[key] = trial_pairs
                    if not self._assignment_memory_feasible(trial):
                        continue
                    old_forwarding = self._subtask_pair_spread_cost(pairs)
                    new_forwarding = self._subtask_pair_spread_cost(trial_pairs)
                    estimated_saving = old_forwarding - new_forwarding
                    candidates.append(
                        (
                            -estimated_saving,
                            self._assignment_label(key),
                            old_pair,
                            new_pair,
                            trial,
                        )
                    )
        candidates.sort()
        for _, label, old_pair, new_pair, trial in candidates[:max_candidates]:
            yield {
                "strategy": "relocate_same_expert",
                "subtask": label,
                "old_pair": list(old_pair),
                "new_pair": list(new_pair),
                "assignments": trial,
            }

    def _subtask_pair_spread_cost(self, pairs: Sequence[tuple[ServerId, ExpertId]]) -> float:
        servers = sorted({server_id for server_id, _ in pairs})
        total = 0.0
        for index, source_server in enumerate(servers):
            for target_server in servers[index + 1 :]:
                total += self.evaluator.wired_weight(source_server, target_server)
        return total

    def _assignment_memory_feasible(self, assignments) -> bool:
        _, used_memory = self._activation_state_from_assignments(assignments)
        return all(
            used_memory.get(server_id, 0.0) <= self.servers[server_id].gpu_memory + 1e-12
            for server_id in self.servers
        )
    def _selection_metadata_from_assignments(self, assignments):
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        subtask_by_key = self._subtask_by_key()
        for key, pairs in assignments.items():
            selected_experts = {expert_id for _, expert_id in pairs}
            selected_experts_by_key[key] = selected_experts
            subtask = subtask_by_key.get(key)
            if subtask is not None:
                selected_probability[self._assignment_label(key)] = self.evaluator.selection_probability(
                    subtask,
                    selected_experts,
                )
        return selected_probability, selected_experts_by_key

    def _activation_state_from_assignments(self, assignments):
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        for pairs in assignments.values():
            for server_id, expert_id in pairs:
                activated.setdefault(server_id, set()).add(expert_id)
        used_memory = {
            server_id: sum(self.experts[expert_id].memory for expert_id in experts if expert_id in self.experts)
            for server_id, experts in activated.items()
        }
        return activated, used_memory

    def _subtask_by_key(self) -> Dict[AssignmentKey, SubtaskSpec]:
        return {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }

    def _feasible_servers_for_replacement(self, expert_id: ExpertId) -> list[ServerId]:
        expert = self.experts[expert_id]
        feasible: list[ServerId] = []
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            if expert.memory <= server.gpu_memory:
                feasible.append(server_id)
        return sorted(feasible)

    def _post_prune_groups(
        self,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        selected_probability: Mapping[str, float],
        groups: Sequence[GroupSpec],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
    ) -> tuple[list[GroupSpec], Dict[Tuple[str, str, str], Set[str]]]:
        current_groups = list(groups)
        current_features = {key: set(value) for key, value in subtask_features.items()}
        current_eval = self._evaluate_prune_candidate(assignments, current_groups, current_features)
        if current_eval is None:
            return current_groups, current_features
        current_cost = current_eval.objective.total_cost
        self.post_pruning_report.update(
            {
                "removed_groups": 0,
                "cost_before": current_cost,
                "cost_after": current_cost,
            }
        )
        if current_eval.violations:
            return current_groups, current_features

        removed = 0
        candidates = sorted(
            current_groups,
            key=lambda group: self._group_prune_priority(group, assignments),
            reverse=True,
        )
        for group in candidates:
            group_key = (group.server_id, group.id)
            if not any((item.server_id, item.id) == group_key for item in current_groups):
                continue
            trial_groups = [item for item in current_groups if (item.server_id, item.id) != group_key]
            trial_features = self._served_subtask_features(assignments, current_features, trial_groups)
            trial_eval = self._evaluate_prune_candidate(assignments, trial_groups, trial_features)
            if trial_eval is None or trial_eval.violations:
                continue
            trial_cost = trial_eval.objective.total_cost
            if trial_cost <= current_cost + 1e-9:
                current_groups = trial_groups
                current_features = trial_features
                current_cost = trial_cost
                removed += 1

        self.post_pruning_report.update(
            {
                "removed_groups": removed,
                "cost_after": current_cost,
            }
        )
        return current_groups, current_features

    def _served_subtask_features(
        self,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        requested_features: Mapping[Tuple[str, str, str], Set[str]],
        groups: Sequence[GroupSpec],
    ) -> Dict[Tuple[str, str, str], Set[str]]:
        available_features: Set[str] = set()
        for group in groups:
            available_features.update(group.required_features)
        served: Dict[Tuple[str, str, str], Set[str]] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                required = set(subtask.required_features)
                for server_id in self.evaluator.participating_servers(assignments, key):
                    feature_key = self._server_feature_key(key, server_id)
                    desired = set(requested_features.get(feature_key, set()))
                    kept = desired & required & available_features
                    if kept:
                        served[feature_key] = kept
        return served

    def _evaluate_prune_candidate(
        self,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
    ):
        (
            feature_source_map,
            backhaul,
            _,
            resolved_features,
            missing,
        ) = self._far_resolve_feature_sources(
            assignments,
            groups,
            requested_subtask_features=subtask_features,
            allow_new_backhaul=True,
            preserve_base_backhaul=False,
        )
        if missing:
            return None
        data_req = self._subtask_data_requirements(assignments, resolved_features)
        feature_to_groups = self._feature_to_groups(groups)
        try:
            resolved_groups, _ = self._derive_group_bandwidths(
                groups,
                assignments,
                data_req,
                backhaul,
                feature_to_groups,
                feature_source_map=feature_source_map,
                subtask_features=resolved_features,
            )
        except InfeasibleBandwidthBudget:
            return None
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        return evaluator.evaluate(
            assignments,
            resolved_groups,
            backhaul,
            subtask_features=resolved_features,
            feature_source_map=feature_source_map,
        )

    def _group_prune_priority(
        self,
        group: GroupSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
    ) -> float:
        try:
            budget = self.evaluator.bandwidth_time_budget()
            bandwidth = self.evaluator.derive_group_bandwidth_for_budget(group, budget)
        except Exception:
            bandwidth = self.evaluator.group_feature_volume(group)
        return (self.config.c_bw * bandwidth) + self.config.c_fwd

    def _recompute_loss_metadata(
        self,
        assignments,
        selected_probability,
        groups,
        subtask_features,
    ):
        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                target_servers = self.evaluator.participating_servers(assignments, key)
                probability = selected_probability.get(label, 0.0)
                avg_rec = self._average_reconstruction_for_servers(
                    subtask,
                    key,
                    target_servers,
                    subtask_features,
                    assignments,
                )
                required_probability[label] = self._required_probability_with_reconstruction(subtask, avg_rec)
                reconstruction_loss[label] = avg_rec
                performance_loss[label] = self.evaluator.performance_loss_from_probability_and_reconstruction(
                    probability,
                    avg_rec,
                )
        return list(groups), subtask_features, required_probability, reconstruction_loss, performance_loss
