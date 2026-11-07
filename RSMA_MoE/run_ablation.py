"""Ablation study for RESG-GoT (7 figures).

Variants (all other parts of RESG-GoT unchanged):
    full      RESG-GoT            SCOPE + CFRG + FAR
    no_scope  SCOPE -> Top-K      Top-K expert selection + CFRG + FAR
    no_group  w/o RSMA grouping   SCOPE + CFRG with N_th = 1 (every IoT alone) + FAR
    no_far    w/o FAR             SCOPE + CFRG, no Phase-3 refinement
    no_cfrg   CFRG -> CSSGD       optional (--variants), SCOPE + CSSGD grouping + FAR;
                                  usually infeasible because CSSGD does not collect
                                  the features SCOPE's expert sets need.

Figures (x sweeps reuse run_ten_figures.py settings and seeds):
    (a) Fig 1  Number of tasks                      / Total cost
    (b) Fig 6  Number of subtasks per task          / Total cost
    (c) Fig 10 Number of shared subtasks            / Total cost
    (d) Fig 13 Avg. deadline (from Fig 4 runs)      / Bandwidth cost
    (e) Fig 15 Avg. required features (Fig 9 runs)  / Communication cost
    (f) Fig 12 Number of IoTs (from Fig 2 runs)     / RSMA groups
    (g) Fig 2  Number of IoTs                       / Total cost
Only 6 sweeps are simulated (1, 2, 4, 6, 9, 10); Fig 12/13/15 reuse them.

Every attempt runs all variants.  An attempt is accepted when RESG-GoT
(full) is feasible, i.e. the same acceptance as the main experiment; each
variant's feasibility is recorded.  Plotting (default) averages each variant
over the accepted attempts where it is feasible; --paired keeps only attempts
where every variant is feasible.  feasibility.csv lists the feasible ratio.

Usage (in the folder with run_ten_figures.py):
    py run_ablation.py                          # default setting only (160 attempts, first 100 accepted)
    py run_ablation.py --attempts 160 --runs 100 --workers 8
    py run_ablation.py --sweep                  # 7-figure x sweeps, 100 accepted runs per x
    py run_ablation.py --sweep --resume results\\ablation\\<dir>   # continue a sweep run
    py run_ablation.py --plot-only results\\ablation\\<dir> [--paired]
Output: results/ablation/<timestamp>/figXX_<key>/observations.csv and
        figures_nolegend/ (PNG, same style as replot_no_legend.py).
This script does not modify any existing file.
"""
from __future__ import annotations

import argparse
import csv
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import run_ten_figures as RT  # noqa: E402

OUTPUT_ROOT = BASE_DIR / "results" / "ablation"
SWEEPS = (1, 2, 4, 6, 9, 10)            # simulated
PLOTS = (1, 6, 10, 13, 15, 12, 2)       # (a)-(g)
VARIANTS = ("full", "no_scope", "no_group", "no_far")
LABELS = {
    "full": "RESG-GoT",
    "no_scope": r"SCOPE$\rightarrow$Top-$K$",
    "no_group": "w/o RSMA grouping",
    "no_far": "w/o FAR",
    "no_cfrg": r"CFRG$\rightarrow$CSSGD",
    "no_cfrg_all": r"CFRG$\rightarrow$CSSGD (all)",
    "cfrg_loc": r"CFRG$\rightarrow$Location",
    "cfrg_loc_all": r"CFRG$\rightarrow$Location (all)",
}
STYLES = {  # (color, marker, linestyle)
    "full": ("royalblue", "^", ":"),
    "no_scope": ("darkorange", "s", "--"),
    "no_group": ("limegreen", "o", "-"),
    "no_far": ("crimson", "D", "-."),
    "no_cfrg": ("darkviolet", "v", (0, (3, 1, 1, 1, 1, 1))),
    "no_cfrg_all": ("gray", "P", "-"),
    "cfrg_loc": ("goldenrod", "X", "--"),
    "cfrg_loc_all": ("teal", "*", "-."),
}
PLAIN = {"full": "RESG-GoT", "no_scope": "SCOPE->Top-K", "no_group": "w/o RSMA grouping",
         "no_far": "w/o FAR", "no_cfrg": "CFRG->CSSGD",
         "no_cfrg_all": "CFRG->CSSGD (all groups)", "cfrg_loc": "CFRG->Location-aware",
         "cfrg_loc_all": "CFRG->Location-aware (all groups)"}
OUR_FLAG = "run_our_similarity_aware_topk_backhaul"
OUR_KEY = OUR_FLAG[len("run_"):]
FIELDS = RT.FIELDS + ("activated_experts", "feasible", "violation")
ATTEMPT_FIELDS = ("x", "attempt", "accepted", "reason", "seconds")
# CFRG -> CSSGD: True = every CSSGD candidate group uploads (full feature
# coverage, the role CFRG plays); False = groups only when loss needs them.
CSSGD_ALL_GROUPS = False


# --------------------------------------------------------------------------- variants
def _cssgd_phase2(self, assignments, selected_probability, selected_experts_by_key,
                  activated, used_memory):
    """Replace CFRG with the CSSGD (GSSGD DBG) grouping used by the baselines.

    Same post-processing as the Top-K+CSSGD baseline: CSSGD builds candidate
    groups under every server, groups are activated only when a subtask's
    loss exceeds its threshold, and missing features are fetched by backhaul.
    """
    from compared_method.hybrid_topk_gssgd_backhaul import HybridTopKGSSGDBackhaulPipeline
    from compared_method.hybrid_topk_location_aware_backhaul import ChainedPipelineResult
    from utils.formulation import FormulationEvaluator

    candidate_features = self._candidate_server_features(assignments)
    if getattr(self, "_ablation_grouping", "cssgd") == "location":
        # Location-aware: k-means on IoT positions under each server (the baselines' method)
        from compared_method.hybrid_topk_location_aware_backhaul import ChainedComparisonPipeline
        candidate_groups = ChainedComparisonPipeline._build_distance_groups(self, candidate_features)
    else:
        candidate_groups = HybridTopKGSSGDBackhaulPipeline._build_distance_groups(self, candidate_features)
    all_groups = getattr(self, "_ablation_cssgd_all_groups", CSSGD_ALL_GROUPS)
    activate = self._use_all_candidate_groups if all_groups else self._repair_loss_with_groups
    active_groups, subtask_features, required_prob, rec_loss, perf_loss = activate(
        assignments, selected_probability, selected_experts_by_key,
        candidate_groups, activated, used_memory,
    )
    data_req = self._subtask_data_requirements(assignments, subtask_features)
    server_features = self._server_required_features(data_req)
    feature_to_groups = self._feature_to_groups(active_groups)
    backhaul, plan = self._derive_backhaul(data_req, feature_to_groups)
    groups, budgets = self._derive_group_bandwidths(
        active_groups, assignments, data_req, backhaul, feature_to_groups,
        subtask_features=subtask_features,
    )
    # Explicit feature sources so that FAR can refine this state.
    feature_source_map = {}
    for task in self.tasks.values():
        for subtask in task.subtasks:
            key = (task.id, subtask.id)
            for server_id in self.evaluator.participating_servers(assignments, key):
                for feature in self.evaluator.features_for_server(subtask_features, key, server_id):
                    group = self.evaluator.serving_group_for_feature(
                        feature, server_id, feature_to_groups, backhaul)
                    if group is not None:
                        feature_source_map[(task.id, subtask.id, server_id, feature)] = (group.server_id, group.id)
    evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks,
                                     replace(self.config, derive_bandwidth=False))
    evaluation = evaluator.evaluate(assignments, groups, backhaul, subtask_features=subtask_features)
    violations = [*self.scheduler_violations, *evaluation.violations]
    self.phase2_report = {"mode": f"{getattr(self, '_ablation_grouping', 'cssgd')} (ablation)"}
    return ChainedPipelineResult(
        expert_placement=self._activated_expert_map(assignments),
        subtask_assignment=assignments,
        subtask_data_requirements=data_req,
        server_required_features={sid: sorted(v) for sid, v in server_features.items()},
        rsma_groups=self._group_map(groups),
        group_required_features={self._group_label(g): sorted(g.required_features) for g in groups},
        group_time_budgets={self._group_key_label(k): v for k, v in budgets.items()},
        rsma_group_bandwidths=self._group_bandwidth_map(groups),
        backhaul_plan=plan,
        selected_probability=selected_probability,
        required_probability=required_prob,
        reconstruction_loss=rec_loss,
        performance_loss=perf_loss,
        total_cost=evaluation.objective.total_cost,
        activation_cost=evaluation.objective.activation_cost,
        bandwidth_cost=evaluation.objective.bandwidth_cost,
        forwarding_cost=evaluation.objective.forwarding_cost,
        inference_cost=evaluation.objective.inference_cost,
        backhaul=backhaul,
        violations=violations,
        evaluation=evaluation,
        feature_source_map=feature_source_map,
    )


def _no_far(self, assignments, current_result):
    self.phase3_report = {"mode": "skipped (ablation)"}
    return current_result


def _variant_patches(variant):
    from our_alg.similarity_aware_topk_backhaul import SimilarityAwareTopKBackhaulPipeline as P
    from compared_method.hybrid_topk_location_aware_backhaul import ChainedComparisonPipeline
    if variant == "no_scope":
        return P, {"_assign_topk_experts": ChainedComparisonPipeline._assign_topk_experts}
    if variant == "no_cfrg":
        return P, {"_run_phase2_for_assignments": _cssgd_phase2, "_ablation_cssgd_all_groups": False}
    if variant == "no_cfrg_all":
        return P, {"_run_phase2_for_assignments": _cssgd_phase2, "_ablation_cssgd_all_groups": True}
    if variant == "cfrg_loc":
        return P, {"_run_phase2_for_assignments": _cssgd_phase2, "_ablation_cssgd_all_groups": False,
                   "_ablation_grouping": "location"}
    if variant == "cfrg_loc_all":
        return P, {"_run_phase2_for_assignments": _cssgd_phase2, "_ablation_cssgd_all_groups": True,
                   "_ablation_grouping": "location"}
    if variant == "no_far":
        return P, {"_phase3_refine": _no_far}
    return P, {}


def run_variant(variant, config, graphs, seeds):
    from experiment_runner import run_enabled_methods
    cls, patches = _variant_patches(variant)
    saved = {name: cls.__dict__.get(name) for name in patches}
    try:
        for name, func in patches.items():
            setattr(cls, name, func)
        only = replace(config, **{flag: flag == OUR_FLAG for flag in RT.METHOD_FLAGS})
        if variant == "no_group":
            only = replace(only, max_group_size=1)
        return run_enabled_methods(only, graphs, seeds, print_details=False, show_plot=False)[OUR_KEY]
    finally:
        for name, func in saved.items():
            if func is None:
                delattr(cls, name)
            else:
                setattr(cls, name, func)


# --------------------------------------------------------------------------- simulation
def _point_files(run_dir, figure, value):
    fig_dir = RT._figure_dir(run_dir, figure)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{float(value):g}"
    return fig_dir / f"obs_x{tag}.csv", fig_dir / f"att_x{tag}.csv"


def _progress(obs_path, att_path):
    accepted, last = 0, 0
    if att_path.exists():
        with att_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                last = max(last, int(row["attempt"]))
                accepted += row["accepted"] == "True"
    return accepted, last


def run_point(job):
    """All attempts for one (figure, x) until `runs` attempts are accepted."""
    number, value, runs, max_attempts, run_dir_text, variants = job
    from experiment_runner import generate_task_graphs, resolve_run_seeds
    run_dir = Path(run_dir_text)
    figure = RT.FIGURE_BY_NUMBER[number]
    obs_path, att_path = _point_files(run_dir, figure, value)
    accepted, attempt = _progress(obs_path, att_path)
    base = RT._base_config(RT._figure_dir(run_dir, figure))
    config = figure.apply(base, value)
    new_obs, new_att = not obs_path.exists(), not att_path.exists()
    with obs_path.open("a", newline="", encoding="utf-8") as of, \
            att_path.open("a", newline="", encoding="utf-8") as af:
        ow = csv.DictWriter(of, fieldnames=FIELDS, extrasaction="ignore")
        aw = csv.DictWriter(af, fieldnames=ATTEMPT_FIELDS)
        if new_obs:
            ow.writeheader()
        if new_att:
            aw.writeheader()
        while accepted < runs and attempt < max_attempts:
            attempt += 1
            start = time.time()
            seeds = resolve_run_seeds(config, run_index=attempt - 1)
            rows, reason = [], ""
            try:
                graphs = generate_task_graphs(config, seeds, verbose=False, export_artifacts=False)
            except Exception as error:
                graphs, reason = None, f"graph error: {str(error)[:120]}"
            if graphs is not None:
                for variant in variants:
                    try:
                        result = run_variant(variant, config, graphs, seeds)
                        violations = list(getattr(result, "violations", None) or [])
                    except Exception as error:  # e.g. negative uplink budget
                        result, violations = None, [f"error: {error}"]
                    if variant == "full" and violations:
                        reason = f"full: {str(violations[0])[:120]}"
                        break  # same acceptance rule as the main experiment
                    row = RT._record(result) if result is not None else {}
                    row.update(
                        figure=number, key=figure.key, x=value, attempt=attempt,
                        dag_seed=getattr(seeds, "dag_seed", ""),
                        network_seed=getattr(seeds, "network_seed", ""), method_key=variant,
                        activated_experts=(
                            len({p for pairs in result.subtask_assignment.values() for p in pairs})
                            if result is not None else ""),
                        feasible=not violations,
                        violation=str(violations[0])[:150] if violations else "",
                    )
                    rows.append(row)
            ok = not reason
            if ok:
                accepted += 1
                for row in rows:
                    row["accepted_index"] = accepted
                    ow.writerow(row)
                of.flush()
            aw.writerow({"x": value, "attempt": attempt, "accepted": ok, "reason": reason,
                         "seconds": round(time.time() - start, 2)})
            af.flush()
    return f"Fig {number} x={value:g}: {accepted}/{runs} accepted after {attempt} attempts"


def merge(run_dir, variants, paired):
    """Write observations.csv (feasible rows only) for the plotter and feasibility.csv."""
    summary = []
    for number in SWEEPS:
        figure = RT.FIGURE_BY_NUMBER[number]
        fig_dir = RT._figure_dir(run_dir, figure)
        rows = []
        for part in sorted(fig_dir.glob("obs_x*.csv")):
            with part.open(encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
        if not rows:
            continue
        by_attempt = {}
        for row in rows:
            by_attempt.setdefault((row["x"], row["attempt"]), {})[row["method_key"]] = row
        keep = []
        for (x, attempt), runs in by_attempt.items():
            feasible = {v for v, r in runs.items() if r["feasible"] == "True"}
            if paired and not set(variants) <= feasible:
                continue
            keep.extend(runs[v] for v in variants if v in feasible)
        with (fig_dir / "observations.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(keep)
        for x in sorted({k[0] for k in by_attempt}, key=float):
            attempts = [runs for (xx, _), runs in by_attempt.items() if xx == x]
            entry = {"figure": number, "key": figure.key, "x": x, "accepted_attempts": len(attempts)}
            for v in variants:
                ok = sum(1 for runs in attempts if runs.get(v, {}).get("feasible") == "True")
                entry[f"{v}_feasible"] = ok
                entry[f"{v}_ratio"] = round(ok / len(attempts), 3) if attempts else ""
            entry["all_feasible"] = sum(
                1 for runs in attempts if all(runs.get(v, {}).get("feasible") == "True" for v in variants))
            summary.append(entry)
    if summary:
        with (run_dir / "feasibility.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Feasibility per x: {run_dir / 'feasibility.csv'}")


# --------------------------------------------------------------------------- plotting
def plot(run_dir, plot_runs, variants=VARIANTS, paired=False):
    merge(run_dir, variants, paired)
    RT.METHOD_ORDER = tuple(variants)
    RT.METHOD_STYLE = dict(STYLES)
    RT.PLOT_LABELS = dict(LABELS)
    RT.METHOD_DISPLAY_NAMES = dict(LABELS)
    script = BASE_DIR / "replot_no_legend.py"
    if not script.exists():
        raise SystemExit("replot_no_legend.py not found next to run_ablation.py")
    sys.argv = [str(script), str(run_dir), "--figures", ",".join(map(str, PLOTS)),
                "--plot-runs", str(plot_runs)]
    runpy.run_path(str(script), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=100, help="accepted runs per x value (default 100)")
    parser.add_argument("--max-attempts", type=int, default=2000, help="attempt limit per x value")
    parser.add_argument("--workers", type=int, default=0, help="parallel processes (default: CPU cores - 1)")
    parser.add_argument("--figures", default=",".join(map(str, SWEEPS)),
                        help="sweeps to simulate, subset of 1,2,4,6,9,10")
    parser.add_argument("--resume", type=Path, default=None, help="continue this run directory")
    parser.add_argument("--plot-only", type=Path, default=None, help="only redraw figures")
    parser.add_argument("--plot-runs", type=int, default=300, help="runs averaged per x when plotting")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="variants to run/plot (full must be first); add no_cfrg for CFRG->CSSGD")
    parser.add_argument("--paired", action="store_true",
                        help="plot only attempts where every variant is feasible")
    parser.add_argument("--default", action="store_true",
                        help="run only the default setting (no x sweep); this is also what happens "
                             "when no option is given")
    parser.add_argument("--sweep", action="store_true",
                        help="run the 7-figure x sweeps instead of the default setting")
    parser.add_argument("--attempts", type=int, default=160,
                        help="--default: attempts to run (about 70%% are accepted)")
    parser.add_argument("--summary-only", type=Path, default=None,
                        help="--default: only rebuild default_summary.csv from a run directory")
    args = parser.parse_args()
    variants = tuple(v.strip() for v in args.variants.split(","))
    if variants[0] != "full" or not set(variants) <= set(LABELS):
        raise SystemExit(f"--variants must start with full and use {sorted(LABELS)}")

    if not args.sweep and not args.plot_only:
        args.default = True
    if args.default:
        if args.summary_only:
            summarize_default(args.summary_only, variants, args.runs)
        else:
            main_default(args, variants)
        return
    if args.plot_only:
        plot(args.plot_only, args.plot_runs, variants, args.paired)
        return
    run_dir = args.resume or OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    sweeps = [int(s) for s in args.figures.split(",")]
    jobs = [(n, float(v), args.runs, args.max_attempts, str(run_dir), variants)
            for n in sweeps for v in RT.FIGURE_BY_NUMBER[n].values]
    workers = args.workers or max(1, min((os.cpu_count() or 2) - 1, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run_point, job) for job in jobs]):
            print(future.result(), flush=True)
    plot(run_dir, args.plot_runs, variants, args.paired)


# --------------------------------------------------------------------------- default point only
DEFAULT_FIELDS = FIELDS


def run_default_chunk(job):
    """Attempts `attempts` at the default setting (no x sweep)."""
    attempts, out_text, variants = job
    from experiment_runner import generate_task_graphs, resolve_run_seeds
    out = Path(out_text)
    config = RT._base_config(out)
    path = out / f"default_{attempts[0]:05d}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS + ("accepted",), extrasaction="ignore")
        writer.writeheader()
        for attempt in attempts:
            seeds = resolve_run_seeds(config, run_index=attempt - 1)
            try:
                graphs = generate_task_graphs(config, seeds, verbose=False, export_artifacts=False)
            except Exception:
                continue
            rows, accepted = [], True
            for variant in variants:
                try:
                    result = run_variant(variant, config, graphs, seeds)
                    violations = list(getattr(result, "violations", None) or [])
                except Exception as error:
                    result, violations = None, [f"error: {error}"]
                if variant == "full" and violations:
                    accepted = False
                    break
                row = RT._record(result) if result is not None else {}
                row.update(figure=0, key="default", x=0, attempt=attempt, method_key=variant,
                           activated_experts=(len({p for pairs in result.subtask_assignment.values()
                                                   for p in pairs}) if result is not None else ""),
                           feasible=not violations,
                           violation=str(violations[0])[:150] if violations else "", accepted=True)
                rows.append(row)
            if accepted:
                writer.writerows(rows)
                f.flush()
    return f"attempts {attempts[0]}-{attempts[-1]} done"


def summarize_default(out, variants, runs):
    rows = []
    for part in sorted(out.glob("default_0*.csv")):
        with part.open(encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    attempts = sorted({int(r["attempt"]) for r in rows})[:runs]
    by = {}
    for r in rows:
        if int(r["attempt"]) in attempts:
            by.setdefault(int(r["attempt"]), {})[r["method_key"]] = r
    with (out / "default_observations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(r for a in attempts for r in by[a].values())
    metrics = ("total_cost", "activation_cost", "inference_cost", "bandwidth_cost",
               "forwarding_cost", "num_groups", "num_iots_used", "activated_experts")
    table = []
    for label, subset in (("own feasible", None), ("all feasible", "all")):
        for v in variants:
            sel = [by[a][v] for a in attempts if v in by[a] and by[a][v]["feasible"] == "True"
                   and (subset is None or all(by[a].get(u, {}).get("feasible") == "True" for u in variants))]
            entry = {"average_over": label, "variant": PLAIN[v], "runs": len(sel),
                     "feasible_ratio": round(sum(1 for a in attempts if by[a].get(v, {}).get("feasible") == "True")
                                             / max(len(attempts), 1), 3)}
            for m in metrics:
                vals = [float(r[m]) for r in sel if r.get(m) not in ("", None)]
                # forwarding_cost (wired transfers between servers) is called backhaul cost in the paper
                entry["backhaul_cost" if m == "forwarding_cost" else m] = (
                    round(sum(vals) / len(vals), 2) if vals else "")
            table.append(entry)
    with (out / "default_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"{len(attempts)} accepted attempts (RESG-GoT feasible)")
    for e in table:
        print(f"{e['average_over']:13s} {e['variant']:22s} runs {e['runs']:4d} feasible {e['feasible_ratio']:.0%}  "
              + "  ".join(f"{'backhaul' if m == 'forwarding_cost' else m.split('_')[0]} {e['backhaul_cost' if m == 'forwarding_cost' else m]}"
                          for m in metrics))
    print(f"Saved: {out / 'default_summary.csv'}")


def main_default(args, variants):
    out = args.resume or OUTPUT_ROOT / ("default_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out}")
    done = set()
    for part in out.glob("default_0*.csv"):
        with part.open(encoding="utf-8") as f:
            done.update(int(r["attempt"]) for r in csv.DictReader(f))
    start = max(done, default=0) + 1
    todo = list(range(start, start + args.attempts))
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    size = max(1, -(-len(todo) // (workers * 4)))
    chunks = [todo[i:i + size] for i in range(0, len(todo), size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run_default_chunk, (c, str(out), variants)) for c in chunks]):
            print(future.result(), flush=True)
    summarize_default(out, variants, args.runs)


if __name__ == "__main__":
    main()
