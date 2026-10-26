from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.formulation import AssignmentKey, ExpertId, ServerId, SubtaskSpec


class Phase1ServerExpertSelectionMixin:
    """Phase I: SCOPE shared-structure placement and expert activation."""

    def _assign_topk_experts(self):
        """Run SCOPE while retaining the legacy pipeline entry-point name."""
        context = self._scope_context()
        placement: Dict[AssignmentKey, ServerId] = {}
        structures: list[dict[str, Any]] = []
        placement_decisions: list[dict[str, Any]] = []

        shared_types = set(context["shared_types"])
        unprocessed = set(shared_types)
        while unprocessed:
            seed = max(
                unprocessed,
                key=lambda node_type: (
                    context["ordering_index"][node_type],
                    node_type,
                ),
            )
            node_types = [seed]
            support = set(context["support_by_type"][seed])
            structure_edges: Set[Tuple[str, str]] = set()
            provisional, decisions = self._scope_place_structure(
                node_types,
                support,
                placement,
                context,
            )
            placement_decisions.extend(decisions)
            unprocessed.remove(seed)
            expansion_log: list[dict[str, Any]] = []

            while True:
                candidates = self._scope_frontier(node_types, support, unprocessed, context)
                if not candidates:
                    break

                evaluated = []
                for candidate in candidates:
                    candidate_support, candidate_edges = self._scope_expansion_support(
                        node_types,
                        support,
                        candidate,
                        context,
                    )
                    gain = (
                        (len(node_types) + 1) * len(candidate_support)
                        - len(node_types) * len(support)
                    )
                    evaluated.append(
                        (
                            gain,
                            len(candidate_support),
                            context["ordering_index"].get(candidate, 0),
                            candidate,
                            candidate_support,
                            candidate_edges,
                        )
                    )

                best = max(evaluated, key=lambda item: item[:4])
                gain, support_count, _, candidate, candidate_support, candidate_edges = best
                accepted = support_count >= 2 and gain >= 0
                expansion_log.append(
                    {
                        "candidate": candidate,
                        "remaining_support": sorted(candidate_support),
                        "support_count": support_count,
                        "expansion_gain": gain,
                        "accepted": accepted,
                    }
                )
                if not accepted:
                    break

                node_types.append(candidate)
                support = set(candidate_support)
                structure_edges.update(candidate_edges)
                provisional, decisions = self._scope_place_structure(
                    node_types,
                    support,
                    placement,
                    context,
                )
                placement_decisions.extend(decisions)
                unprocessed.remove(candidate)

            for node_type in node_types:
                server_id = provisional[node_type]
                for task_id in support:
                    key = context["occurrence_by_task_type"].get((task_id, node_type))
                    if key is not None:
                        placement[key] = server_id

            structures.append(
                {
                    "id": f"R{len(structures)}",
                    "node_types": list(node_types),
                    "edges": [list(edge) for edge in sorted(structure_edges)],
                    "support_tasks": sorted(support),
                    "support_count": len(support),
                    "placement_by_type": dict(sorted(provisional.items())),
                    "expansions": expansion_log,
                }
            )

        remaining = set(context["subtask_by_key"]) - set(placement)
        while remaining:
            ranked = []
            for key in remaining:
                server_id, score, components = self._scope_choose_server(
                    [key],
                    placement,
                    context,
                )
                ranked.append((score, key, server_id, components))
            score, key, server_id, components = max(
                ranked,
                key=lambda item: (item[0], item[1], item[2]),
            )
            placement[key] = server_id
            placement_decisions.append(
                self._scope_placement_record(
                    deployment_unit=[key],
                    node_type=context["type_by_key"][key],
                    server_id=server_id,
                    score=score,
                    components=components,
                    shared=False,
                )
            )
            remaining.remove(key)

        activated, used_memory, activation_report = self._scope_activate_experts(placement, context)
        assignments: Dict[AssignmentKey, list[tuple[ServerId, ExpertId]]] = {}
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        unsatisfied = 0
        total_pairs = 0

        for key, subtask in context["subtask_by_key"].items():
            server_id = placement[key]
            expert_ids = sorted(activated.get(server_id, set()))
            pairs = [(server_id, expert_id) for expert_id in expert_ids]
            assignments[key] = pairs
            selected_experts_by_key[key] = set(expert_ids)
            probability = self.evaluator.selection_probability(subtask, expert_ids)
            selected_probability[self._assignment_label(key)] = probability
            threshold = self.evaluator.conformal_loss_threshold(subtask)
            full_feature_loss = self.evaluator.performance_loss_from_probability(
                subtask,
                probability,
                set(subtask.required_features),
            )
            if not pairs:
                self.scheduler_violations.append(
                    f"SCOPE {key}: server {server_id} has no activated expert"
                )
            if full_feature_loss > threshold + 1e-12:
                unsatisfied += 1
                self.scheduler_violations.append(
                    f"SCOPE {key}: full-feature loss {full_feature_loss:.6g} "
                    f"> threshold {threshold:.6g}"
                )
            total_pairs += len(pairs)

        self.phase1_report.update(
            {
                "mode": "SCOPE-reuse-aware",
                "placement_rule": (
                    "H_bar * GPU * (1 + max_gating_cosine + dynamic_feature_reuse) "
                    "/ ((epsilon + network_weight_sum) * (1 + dependency_hops))"
                ),
                "feature_reuse_source": (
                    "dynamic union of required features from nodes already placed "
                    "on the candidate server"
                ),
                "expert_reuse_source": (
                    "maximum gating-vector cosine similarity to a node already "
                    "placed on the candidate server, restricted to stored experts"
                ),
                "activation_rule": "unchanged original greedy memory fill",
                "node_type_source": "SubtaskSpec.unique_id (fallback: subtask id)",
                "occurrence_count": len(context["subtask_by_key"]),
                "node_type_count": len(context["occurrences_by_type"]),
                "shared_node_types": sorted(shared_types),
                "ordering_index": dict(sorted(context["ordering_index"].items())),
                "shared_structures": structures,
                "occurrence_placement": {
                    self._assignment_label(key): server_id
                    for key, server_id in sorted(placement.items())
                },
                "placement_decisions": placement_decisions,
                "expert_activation": activation_report,
                "unsatisfied_after_phase1": unsatisfied,
                "average_pairs_per_node": (
                    total_pairs / len(assignments) if assignments else 0.0
                ),
                "assignment_rule": (
                    "X[i,j,s,p] = 1{rho[i,j]=s} * Y[s,p]; every expert "
                    "activated by GreedyFill is initially assigned to every subtask on that server"
                ),
            }
        )
        return assignments, selected_probability, selected_experts_by_key, activated, used_memory

    def _scope_context(self) -> dict[str, Any]:
        subtask_by_key: Dict[AssignmentKey, SubtaskSpec] = {}
        type_by_key: Dict[AssignmentKey, str] = {}
        occurrences_by_type: Dict[str, list[AssignmentKey]] = defaultdict(list)
        occurrence_by_task_type: Dict[Tuple[str, str], AssignmentKey] = {}
        neighbors: Dict[AssignmentKey, Set[AssignmentKey]] = defaultdict(set)
        directed_type_edges_by_task: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        for task in self.tasks.values():
            by_id = {subtask.id: subtask for subtask in task.subtasks}
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                node_type = str(subtask.unique_id or subtask.id)
                subtask_by_key[key] = subtask
                type_by_key[key] = node_type
                occurrences_by_type[node_type].append(key)
                occurrence_by_task_type.setdefault((task.id, node_type), key)
            for subtask in task.subtasks:
                target_key = (task.id, subtask.id)
                for predecessor_id in subtask.predecessors:
                    if predecessor_id not in by_id:
                        continue
                    source_key = (task.id, predecessor_id)
                    neighbors[source_key].add(target_key)
                    neighbors[target_key].add(source_key)
                    directed_type_edges_by_task[task.id].add(
                        (type_by_key[source_key], type_by_key[target_key])
                    )

        support_by_type = {
            node_type: {task_id for task_id, _ in occurrences}
            for node_type, occurrences in occurrences_by_type.items()
        }
        shared_types = {
            node_type
            for node_type, support in support_by_type.items()
            if len(support) >= 2
        }
        shared_neighbors: Dict[str, Set[str]] = {node_type: set() for node_type in shared_types}
        for edges in directed_type_edges_by_task.values():
            for source_type, target_type in edges:
                if source_type in shared_types and target_type in shared_types:
                    shared_neighbors[source_type].add(target_type)
                    shared_neighbors[target_type].add(source_type)
        ordering_index = {
            node_type: len(shared_neighbors[node_type]) + len(support_by_type[node_type])
            for node_type in shared_types
        }
        server_features = {
            server_id: self._server_local_features(
                self._server_transmittable_devices(server_id)
            )
            for server_id in self.servers
        }
        return {
            "subtask_by_key": subtask_by_key,
            "type_by_key": type_by_key,
            "occurrences_by_type": dict(occurrences_by_type),
            "occurrence_by_task_type": occurrence_by_task_type,
            "neighbors": dict(neighbors),
            "directed_type_edges_by_task": dict(directed_type_edges_by_task),
            "support_by_type": support_by_type,
            "shared_types": shared_types,
            "shared_neighbors": shared_neighbors,
            "ordering_index": ordering_index,
            "server_features": server_features,
        }

    def _scope_frontier(
        self,
        node_types: Sequence[str],
        support: Set[str],
        unprocessed: Set[str],
        context: Mapping[str, Any],
    ) -> list[str]:
        frontier = []
        current = set(node_types)
        for candidate in unprocessed:
            for task_id in support:
                edges = context["directed_type_edges_by_task"].get(task_id, set())
                if any(
                    (candidate, node_type) in edges or (node_type, candidate) in edges
                    for node_type in current
                ):
                    frontier.append(candidate)
                    break
        return sorted(frontier)

    def _scope_expansion_support(
        self,
        node_types: Sequence[str],
        support: Set[str],
        candidate: str,
        context: Mapping[str, Any],
    ) -> tuple[Set[str], Set[Tuple[str, str]]]:
        signatures: Dict[Tuple[Tuple[str, str], ...], Set[str]] = defaultdict(set)
        current = set(node_types)
        for task_id in support:
            if (task_id, candidate) not in context["occurrence_by_task_type"]:
                continue
            if any(
                (task_id, node_type) not in context["occurrence_by_task_type"]
                for node_type in current
            ):
                continue
            task_edges = context["directed_type_edges_by_task"].get(task_id, set())
            connecting = tuple(
                sorted(
                    edge
                    for edge in task_edges
                    if (edge[0] == candidate and edge[1] in current)
                    or (edge[1] == candidate and edge[0] in current)
                )
            )
            if connecting:
                signatures[connecting].add(task_id)
        if not signatures:
            return set(), set()
        signature, tasks = max(
            signatures.items(),
            key=lambda item: (len(item[1]), item[0]),
        )
        return set(tasks), set(signature)

    def _scope_place_structure(
        self,
        node_types: Sequence[str],
        support: Set[str],
        committed: Mapping[AssignmentKey, ServerId],
        context: Mapping[str, Any],
    ) -> tuple[Dict[str, ServerId], list[dict[str, Any]]]:
        provisional: Dict[str, ServerId] = {}
        decisions: list[dict[str, Any]] = []
        for node_type in node_types:
            deployment_unit = [
                context["occurrence_by_task_type"][(task_id, node_type)]
                for task_id in sorted(support)
                if (task_id, node_type) in context["occurrence_by_task_type"]
            ]
            local_placement = dict(committed)
            for placed_type, server_id in provisional.items():
                for task_id in support:
                    key = context["occurrence_by_task_type"].get((task_id, placed_type))
                    if key is not None:
                        local_placement[key] = server_id
            server_id, score, components = self._scope_choose_server(
                deployment_unit,
                local_placement,
                context,
            )
            provisional[node_type] = server_id
            decisions.append(
                self._scope_placement_record(
                    deployment_unit=deployment_unit,
                    node_type=node_type,
                    server_id=server_id,
                    score=score,
                    components=components,
                    shared=True,
                )
            )
        return provisional, decisions

    def _scope_choose_server(
        self,
        deployment_unit: Sequence[AssignmentKey],
        placement: Mapping[AssignmentKey, ServerId],
        context: Mapping[str, Any],
    ) -> tuple[ServerId, float, dict[str, float]]:
        candidates = []
        for server_id, server in self.servers.items():
            placed_keys = [
                key
                for key, placed_server in placement.items()
                if placed_server == server_id and key in context["subtask_by_key"]
            ]
            feature_demand: set[str] = set()
            for key in placed_keys:
                feature_demand.update(context["subtask_by_key"][key].required_features)

            expert_score = 0.0
            expert_reuse_score = 0.0
            feature_reuse_score = 0.0
            forwarding_penalty = 0.0
            for key in deployment_unit:
                subtask = context["subtask_by_key"][key]
                expert_score += sum(
                    self.evaluator.expert_contribution(subtask, expert_id)
                    for expert_id in server.stored_experts
                    if expert_id in self.experts
                )
                expert_reuse_score += self._maximum_gating_reuse(
                    subtask,
                    placed_keys,
                    server.stored_experts,
                    context,
                )
                required = set(subtask.required_features)
                feature_reuse_score += (
                    len(required & feature_demand) / len(required)
                    if required
                    else 0.0
                )
                for neighbor in context["neighbors"].get(key, set()):
                    neighbor_server = placement.get(neighbor)
                    if neighbor_server is not None:
                        forwarding_penalty += self.evaluator.wired_weight(
                            server_id,
                            neighbor_server,
                        )

            unit_size = max(len(deployment_unit), 1)
            mean_expert_contribution = expert_score / unit_size
            mean_expert_reuse = expert_reuse_score / unit_size
            mean_feature_reuse = feature_reuse_score / unit_size
            network_weight_sum = sum(
                self.evaluator.wired_weight(server_id, other_server)
                for other_server in self.servers
                if other_server != server_id
            )
            reuse_multiplier = 1.0 + mean_expert_reuse + mean_feature_reuse
            base_index = (
                mean_expert_contribution
                * max(float(server.gpu_memory), 0.0)
                * reuse_multiplier
                / (1e-12 + network_weight_sum)
            )
            placement_index = base_index / (1.0 + forwarding_penalty)
            components = {
                "mean_expert_contribution": mean_expert_contribution,
                "mean_expert_reuse": mean_expert_reuse,
                "mean_feature_reuse": mean_feature_reuse,
                "reuse_multiplier": reuse_multiplier,
                "placed_nodes_on_server": float(len(placed_keys)),
                "dynamic_feature_demand_size": float(len(feature_demand)),
                "gpu_memory": float(server.gpu_memory),
                "network_weight_sum": network_weight_sum,
                "base_index": base_index,
                "forwarding_penalty": forwarding_penalty,
                "placement_index": placement_index,
            }
            candidates.append((placement_index, server_id, components))
        score, server_id, components = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return server_id, score, components

    def _maximum_gating_reuse(
        self,
        subtask: SubtaskSpec,
        placed_keys: Sequence[AssignmentKey],
        stored_experts: set[str],
        context: Mapping[str, Any],
    ) -> float:
        if not placed_keys:
            return 0.0
        indices = sorted(
            {
                self.experts[expert_id].index
                for expert_id in stored_experts
                if expert_id in self.experts and self.experts[expert_id].index >= 0
            }
        )
        if not indices:
            return 0.0
        candidate = self._restricted_gating_vector(subtask, indices)
        return max(
            (
                self._cosine_similarity(
                    candidate,
                    self._restricted_gating_vector(
                        context["subtask_by_key"][placed_key],
                        indices,
                    ),
                )
                for placed_key in placed_keys
            ),
            default=0.0,
        )

    @staticmethod
    def _restricted_gating_vector(
        subtask: SubtaskSpec,
        indices: Sequence[int],
    ) -> tuple[float, ...]:
        return tuple(
            float(subtask.gating_weights[index])
            if index < len(subtask.gating_weights)
            else 0.0
            for index in indices
        )

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))

    def _scope_placement_record(
        self,
        *,
        deployment_unit: Sequence[AssignmentKey],
        node_type: str,
        server_id: ServerId,
        score: float,
        components: Mapping[str, float],
        shared: bool,
    ) -> dict[str, Any]:
        return {
            "node_type": node_type,
            "occurrences": [self._assignment_label(key) for key in deployment_unit],
            "shared_unit": shared,
            "selected_server": server_id,
            "placement_index": score,
            **dict(components),
        }

    def _scope_activate_experts(
        self,
        placement: Mapping[AssignmentKey, ServerId],
        context: Mapping[str, Any],
    ) -> tuple[
        Dict[ServerId, Set[ExpertId]],
        Dict[ServerId, float],
        Dict[ServerId, dict[str, Any]],
    ]:
        activated: Dict[ServerId, Set[ExpertId]] = {server_id: set() for server_id in self.servers}
        used_memory: Dict[ServerId, float] = {server_id: 0.0 for server_id in self.servers}
        report: Dict[ServerId, dict[str, Any]] = {}

        keys_by_server: Dict[ServerId, list[AssignmentKey]] = defaultdict(list)
        for key, server_id in placement.items():
            keys_by_server[server_id].append(key)

        for server_id, server in self.servers.items():
            assigned_keys = keys_by_server.get(server_id, [])
            scores = {}
            for expert_id in server.stored_experts:
                if expert_id not in self.experts:
                    continue
                scores[expert_id] = sum(
                    self.evaluator.expert_contribution(
                        context["subtask_by_key"][key],
                        expert_id,
                    )
                    for key in assigned_keys
                )
            ranked = sorted(
                scores,
                key=lambda expert_id: (-scores[expert_id], expert_id),
            )
            if assigned_keys:
                for expert_id in ranked:
                    memory = max(float(self.experts[expert_id].memory), 0.0)
                    if used_memory[server_id] + memory <= server.gpu_memory + 1e-12:
                        activated[server_id].add(expert_id)
                        used_memory[server_id] += memory
            report[server_id] = {
                "assigned_occurrences": len(assigned_keys),
                "activation_scores": {
                    expert_id: scores[expert_id]
                    for expert_id in ranked
                },
                "activated_experts": sorted(activated[server_id]),
                "used_memory": used_memory[server_id],
                "gpu_memory": float(server.gpu_memory),
            }
        return activated, used_memory, report
