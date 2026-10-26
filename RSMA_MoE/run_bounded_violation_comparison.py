"""Run a bounded five-method comparison with an explicit violation gate.

Every attempt uses the same DAG/network seeds for all five methods. An attempt
is included in the requested 10-run cost statistics only when every method
has exactly zero constraint violations. Rejected attempts are never hidden:
both accepted and rejected attempts are written to JSONL/CSV. The search stops
after 10 accepted attempts or 100 total attempts, whichever happens first.
Complete result JSON files for all five methods are preserved for every
accepted run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from config import ExperimentConfig
from experiment_runner import (
    AverageResult,
    METHOD_DISPLAY_NAMES,
    generate_task_graphs,
    print_average_results,
    resolve_run_seeds,
    result_diagnostics,
    run_enabled_methods,
    show_cost_plot,
)


BASE_DIR = Path(__file__).resolve().parent
COST_FIELDS = (
    "total_cost",
    "activation_cost",
    "bandwidth_cost",
    "forwarding_cost",
    "inference_cost",
)
DIAGNOSTIC_FIELDS = (
    "activation_usage",
    "bandwidth_usage",
    "forwarding_usage",
    "inference_time_ms",
    "avg_servers_per_subtask",
    "avg_experts_per_subtask",
)
METHOD_RESULT_FILENAMES = {
    "hybrid_topk_gssgd_backhaul": "hybrid_topk_gssgd_backhaul_result.json",
    "hybrid_topk_location_aware_backhaul": (
        "hybrid_topk_location_aware_result.json"
    ),
    "wdmoe_gssgd": "wdmoe_gssgd_result.json",
    "wdmoe_location_aware": "wdmoe_location_aware_result.json",
    "our_similarity_aware_topk_backhaul": "jrgep_result.json",
}


def _new_method_totals() -> dict[str, float]:
    return {
        **{field: 0.0 for field in COST_FIELDS},
        **{field: 0.0 for field in DIAGNOSTIC_FIELDS},
        "violation_count": 0.0,
    }


def _method_record(result: Any, config: ExperimentConfig) -> dict[str, Any]:
    diagnostics = result_diagnostics(result, config)
    violations = [str(item) for item in getattr(result, "violations", ())]
    return {
        **{
            field: float(getattr(result, field, 0.0))
            for field in COST_FIELDS
        },
        **{
            field: float(diagnostics[field])
            for field in DIAGNOSTIC_FIELDS
        },
        "violation_count": len(violations),
        "violations": violations,
        "feasible": not violations,
    }


def _add_to_totals(
    totals: dict[str, dict[str, float]],
    method_key: str,
    record: dict[str, Any],
) -> None:
    method_totals = totals.setdefault(method_key, _new_method_totals())
    for field in (*COST_FIELDS, *DIAGNOSTIC_FIELDS, "violation_count"):
        method_totals[field] += float(record[field])


def _averages_from_totals(
    totals: dict[str, dict[str, float]],
    count: int,
) -> dict[str, AverageResult]:
    if count <= 0:
        return {}
    return {
        method_key: AverageResult(
            total_cost=values["total_cost"] / count,
            activation_cost=values["activation_cost"] / count,
            bandwidth_cost=values["bandwidth_cost"] / count,
            forwarding_cost=values["forwarding_cost"] / count,
            inference_cost=values["inference_cost"] / count,
            violation_count=values["violation_count"] / count,
            activation_usage=values["activation_usage"] / count,
            bandwidth_usage=values["bandwidth_usage"] / count,
            forwarding_usage=values["forwarding_usage"] / count,
            inference_time_ms=values["inference_time_ms"] / count,
            avg_servers_per_subtask=(
                values["avg_servers_per_subtask"] / count
            ),
            avg_experts_per_subtask=(
                values["avg_experts_per_subtask"] / count
            ),
        )
        for method_key, values in totals.items()
    }


def _write_progress(
    path: Path,
    *,
    settings: dict[str, Any],
    attempts: int,
    accepted: int,
    rejected: int,
    all_totals: dict[str, dict[str, float]],
    accepted_totals: dict[str, dict[str, float]],
    completed: bool,
) -> None:
    accepted_averages = _averages_from_totals(accepted_totals, accepted)
    payload = {
        "settings": settings,
        "attempt_count": attempts,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "completed": completed,
        "all_attempt_violation_totals": {
            method_key: int(values["violation_count"])
            for method_key, values in all_totals.items()
        },
        "accepted_attempt_violation_totals": {
            method_key: int(values["violation_count"])
            for method_key, values in accepted_totals.items()
        },
        "accepted_method_averages": {
            method_key: asdict(result)
            for method_key, result in accepted_averages.items()
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_accepted_method_results(
    *,
    latest_results_dir: Path,
    accepted_runs_dir: Path,
    accepted_index: int,
    attempt_number: int,
    dag_seed: int,
    network_seed: int,
) -> dict[str, Path]:
    """Preserve complete output JSON files for all methods in an accepted run."""
    run_metadata = {
        "accepted_index": accepted_index,
        "attempt": attempt_number,
        "dag_seed": dag_seed,
        "network_seed": network_seed,
    }
    destination_dir = accepted_runs_dir / (
        f"accepted_{accepted_index:03d}_"
        f"attempt_{attempt_number:04d}_"
        f"dag_{dag_seed}_network_{network_seed}"
    )
    destination_dir.mkdir(parents=True, exist_ok=False)

    saved_paths: dict[str, Path] = {}
    for method_key, source_filename in METHOD_RESULT_FILENAMES.items():
        source_path = latest_results_dir / source_filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Accepted result for {method_key} was not written: "
                f"{source_path}"
            )
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        payload["bounded_run_metadata"] = {
            **run_metadata,
            "method_key": method_key,
            "method_name": METHOD_DISPLAY_NAMES.get(method_key, method_key),
        }
        destination_path = destination_dir / f"{method_key}.json"
        destination_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved_paths[method_key] = destination_path
    return saved_paths


def run_bounded_comparison(
    *,
    target_runs: int = 10,
    max_attempts: int = 100,
    dag_seed: int = 42,
    network_seed: int = 1,
    auto_random_seed: bool = False,
) -> Path:
    if target_runs <= 0:
        raise ValueError("target_runs must be positive")
    if max_attempts < target_runs:
        raise ValueError("max_attempts must be at least target_runs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / "results" / "bounded_comparison" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    method_results_dir = run_dir / "latest_method_results"
    method_results_dir.mkdir(parents=True, exist_ok=True)
    accepted_method_runs_dir = run_dir / "accepted_method_runs"
    accepted_method_runs_dir.mkdir(parents=True, exist_ok=True)

    base = ExperimentConfig()
    config = replace(
        base,
        auto_random_seed=auto_random_seed,
        random_seed=dag_seed,
        network_random_seed=network_seed,
        results_dir=method_results_dir,
        output_dir=run_dir / "latest_generated_dags",
        run_hybrid_topk_gssgd_backhaul=True,
        run_hybrid_topk_location_aware_backhaul=True,
        run_wdmoe_gssgd=True,
        run_wdmoe_location_aware=True,
        run_our_similarity_aware_topk_backhaul=True,
    )
    settings = {
        "target_runs": target_runs,
        "max_attempts": max_attempts,
        "required_violations_per_method": 0,
        "num_dags_per_attempt": config.num_dags,
        "dag_seed": dag_seed,
        "network_seed": network_seed,
        "auto_random_seed": auto_random_seed,
        "acceptance_rule": (
            "all five methods must each have exactly zero violations"
        ),
        "accepted_detailed_results": "accepted_method_runs",
    }
    (run_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    jsonl_path = run_dir / "attempts.jsonl"
    csv_path = run_dir / "attempts_by_method.csv"
    progress_path = run_dir / "progress.json"
    all_totals: dict[str, dict[str, float]] = {}
    accepted_totals: dict[str, dict[str, float]] = {}
    accepted_count = 0
    rejected_count = 0

    csv_fields = (
        "attempt",
        "accepted_index",
        "attempt_accepted",
        "attempt_violation_total",
        "dag_seed",
        "network_seed",
        "method_detail_file",
        "method_key",
        "method_name",
        *COST_FIELDS,
        "violation_count",
        "feasible",
        "violations",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for attempt_index in range(max_attempts):
            attempt_number = attempt_index + 1
            seeds = resolve_run_seeds(config, run_index=attempt_index)
            graphs = generate_task_graphs(
                config,
                seeds,
                verbose=False,
                export_artifacts=False,
            )
            results = run_enabled_methods(
                config,
                graphs,
                seeds,
                print_details=False,
                show_plot=False,
            )
            method_records = {
                method_key: _method_record(result, config)
                for method_key, result in results.items()
            }
            attempt_violation_total = sum(
                record["violation_count"]
                for record in method_records.values()
            )
            attempt_accepted = len(method_records) == 5 and all(
                record["violation_count"] == 0
                for record in method_records.values()
            )
            if attempt_accepted:
                accepted_count += 1
            else:
                rejected_count += 1

            method_detail_paths: dict[str, Path] = {}
            if attempt_accepted:
                method_detail_paths = _save_accepted_method_results(
                    latest_results_dir=method_results_dir,
                    accepted_runs_dir=accepted_method_runs_dir,
                    accepted_index=accepted_count,
                    attempt_number=attempt_number,
                    dag_seed=seeds.dag_seed,
                    network_seed=seeds.network_seed,
                )
            method_detail_files = {
                method_key: str(path.relative_to(run_dir))
                for method_key, path in method_detail_paths.items()
            }

            attempt_record = {
                "attempt": attempt_number,
                "accepted_index": (
                    accepted_count if attempt_accepted else None
                ),
                "accepted": attempt_accepted,
                "attempt_violation_total": attempt_violation_total,
                "dag_seed": seeds.dag_seed,
                "network_seed": seeds.network_seed,
                "method_detail_files": method_detail_files,
                "methods": method_records,
            }
            with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
                jsonl_file.write(
                    json.dumps(attempt_record, ensure_ascii=False) + "\n"
                )

            for method_key, record in method_records.items():
                _add_to_totals(all_totals, method_key, record)
                if attempt_accepted:
                    _add_to_totals(accepted_totals, method_key, record)
                writer.writerow(
                    {
                        "attempt": attempt_number,
                        "accepted_index": (
                            accepted_count if attempt_accepted else ""
                        ),
                        "attempt_accepted": attempt_accepted,
                        "attempt_violation_total": attempt_violation_total,
                        "dag_seed": seeds.dag_seed,
                        "network_seed": seeds.network_seed,
                        "method_detail_file": method_detail_files.get(
                            method_key, ""
                        ),
                        "method_key": method_key,
                        "method_name": METHOD_DISPLAY_NAMES.get(
                            method_key, method_key
                        ),
                        **{field: record[field] for field in COST_FIELDS},
                        "violation_count": record["violation_count"],
                        "feasible": record["feasible"],
                        "violations": " | ".join(record["violations"]),
                    }
                )
            csv_file.flush()

            completed = accepted_count >= target_runs
            _write_progress(
                progress_path,
                settings=settings,
                attempts=attempt_number,
                accepted=accepted_count,
                rejected=rejected_count,
                all_totals=all_totals,
                accepted_totals=accepted_totals,
                completed=completed,
            )
            print(
                f"Attempt {attempt_number}/{max_attempts}: "
                f"violations={attempt_violation_total}, "
                f"accepted={'yes' if attempt_accepted else 'no'}, "
                f"accepted_runs={accepted_count}/{target_runs}"
            )
            if attempt_accepted:
                print(
                    f"  Accepted run {accepted_count}: "
                    f"attempt={attempt_number}, "
                    f"DAG seed={seeds.dag_seed}, "
                    f"network seed={seeds.network_seed}, "
                    f"details={next(iter(method_detail_paths.values())).parent}"
                )
                for method_key, record in method_records.items():
                    method_name = METHOD_DISPLAY_NAMES.get(
                        method_key,
                        method_key,
                    )
                    print(
                        f"    {method_name}: "
                        f"total={record['total_cost']:.6f}, "
                        f"activation={record['activation_cost']:.6f}, "
                        f"bandwidth={record['bandwidth_cost']:.6f}, "
                        f"forwarding={record['forwarding_cost']:.6f}, "
                        f"inference={record['inference_cost']:.6f}, "
                        f"violations={record['violation_count']}"
                    )
            if completed:
                break

    attempt_count = accepted_count + rejected_count
    completed = accepted_count >= target_runs
    accepted_averages = _averages_from_totals(
        accepted_totals,
        accepted_count,
    )
    _write_progress(
        run_dir / "summary.json",
        settings=settings,
        attempts=attempt_count,
        accepted=accepted_count,
        rejected=rejected_count,
        all_totals=all_totals,
        accepted_totals=accepted_totals,
        completed=completed,
    )
    if accepted_averages:
        print_average_results(accepted_averages, config, accepted_count)
        show_cost_plot(
            accepted_averages,
            title=(
                f"Accepted Cost Comparison ({accepted_count} runs; "
                f"attempts={attempt_count})"
            ),
            figure_dir=run_dir / "figure",
            filename=None,
        )

    print(f"\nCompleted: {completed}")
    print(f"Accepted runs: {accepted_count}/{target_runs}")
    print(f"Rejected attempts: {rejected_count}")
    print(f"Total attempts: {attempt_count}/{max_attempts}")
    print(f"Output directory: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-runs", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--dag-seed", type=int, default=42)
    parser.add_argument("--network-seed", type=int, default=1)
    parser.add_argument("--auto-random-seed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bounded_comparison(
        target_runs=args.target_runs,
        max_attempts=args.max_attempts,
        dag_seed=args.dag_seed,
        network_seed=args.network_seed,
        auto_random_seed=args.auto_random_seed,
    )


if __name__ == "__main__":
    main()
