"""
Ablation + Multi-Run Experiment Script

This script implements:
1. 4 Ablation conditions:
   - A: Baseline (task-specific only)
   - B: Baseline + System Wrapper (no extra rules)
   - C: Baseline + Generic Rules (in user prompt)
   - D: Full Improved (system wrapper + generic rules)

2. N=5 repetitions per case per condition
3. Variance tracking + Wilson confidence intervals

Expected runtime: 4-6 hours for full 50 cases × 4 conditions × 5 runs × 2 models
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from scipy import stats

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CANDIDATE_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from ollama_runner import generate, check_ollama_available
from run_eval import (
    load_dataset, run_checks, build_prompt,
    BASELINE_SYSTEM_PROMPT, IMPROVED_SYSTEM_PROMPT
)


# =============================================================================
# Ablation Prompt Conditions
# =============================================================================

CONDITION_A_SYSTEM = "You are a helpful assistant."  # Baseline

CONDITION_B_SYSTEM = """You are a helpful assistant that follows instructions."""  # Minimal wrapper

CONDITION_C_SYSTEM = "You are a helpful assistant."  # Same as baseline (rules in user prompt)

CONDITION_D_SYSTEM = IMPROVED_SYSTEM_PROMPT  # Full improved

# Generic rules for Condition C (append to user prompt)
GENERIC_RULES_SUFFIX = """

RULES:
- Output only what is requested
- For JSON: return valid JSON without markdown
- For citations: use [1], [2] format
- Follow all constraints exactly"""


def get_prompt_for_condition(
    case: dict,
    condition: str  # "A", "B", "C", or "D"
) -> tuple[str, str]:
    """
    Returns (user_prompt, system_prompt) for the given condition.
    
    Conditions:
    A: Baseline (task-specific prompt + minimal system)
    B: Baseline + System Wrapper (adds wrapper, no generic rules)
    C: Baseline + Generic Rules (rules appended to user prompt)
    D: Full Improved (system wrapper + generic rules)
    """
    base_prompt = build_prompt(case, use_improved=False)
    
    if condition == "A":
        return base_prompt, CONDITION_A_SYSTEM
    
    elif condition == "B":
        return base_prompt, CONDITION_B_SYSTEM
    
    elif condition == "C":
        # Append generic rules to user prompt
        user_prompt = base_prompt + GENERIC_RULES_SUFFIX
        return user_prompt, CONDITION_C_SYSTEM
    
    elif condition == "D":
        return base_prompt, CONDITION_D_SYSTEM
    
    else:
        raise ValueError(f"Unknown condition: {condition}")


@dataclass
class AblationResult:
    """Result from one run of one case under one condition."""
    case_id: str
    task_type: str
    condition: str  # "A", "B", "C", "D"
    run_number: int  # 1-5
    model: str
    success: bool
    latency_ms: float
    checks_passed: int
    checks_total: int
    check_details: list
    response_text: str


def run_case_multirun(
    case: dict,
    condition: str,
    model: str,
    n_runs: int = 5
) -> list[AblationResult]:
    """Run a single case N times under one condition."""
    case_id = case.get("id", "unknown")
    task_type = case.get("task_type", "unknown")
    checks = case.get("checks", {})
    
    user_prompt, system_prompt = get_prompt_for_condition(case, condition)
    
    results = []
    for run in range(1, n_runs + 1):
        response = generate(
            user_prompt,
            model,
            system_prompt=system_prompt
        )
        
        if not response.success:
            results.append(AblationResult(
                case_id=case_id,
                task_type=task_type,
                condition=condition,
                run_number=run,
                model=model,
                success=False,
                latency_ms=response.latency_ms,
                checks_passed=0,
                checks_total=1,
                check_details=[{"error": response.error}],
                response_text=""
            ))
            continue
        
        passed, total, details = run_checks(response.text, checks, case)
        
        results.append(AblationResult(
            case_id=case_id,
            task_type=task_type,
            condition=condition,
            run_number=run,
            model=model,
            success=response.success,
            latency_ms=response.latency_ms,
            checks_passed=passed,
            checks_total=total,
            check_details=details,
            response_text=response.text
        ))
    
    return results


def wilson_score_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for proportion.
    Returns (lower_bound, upper_bound).
    """
    if trials == 0:
        return 0.0, 0.0
    
    p_hat = successes / trials
    z = stats.norm.ppf(1 - alpha/2)
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))) / denominator
    
    return max(0, center - margin), min(1, center + margin)


def aggregate_multirun_results(results: list[AblationResult]) -> dict:
    """Aggregate N runs for one case under one condition."""
    if not results:
        return {}
    
    all_pass_count = sum(1 for r in results if r.checks_passed == r.checks_total and r.checks_total > 0)
    n_runs = len(results)
    
    pass_rate = all_pass_count / n_runs if n_runs > 0 else 0
    ci_lower, ci_upper = wilson_score_interval(all_pass_count, n_runs)
    
    latencies = [r.latency_ms for r in results if r.success]
    
    return {
        "case_id": results[0].case_id,
        "condition": results[0].condition,
        "n_runs": n_runs,
        "all_pass_count": all_pass_count,
        "all_pass_rate": pass_rate,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "mean_latency_ms": np.mean(latencies) if latencies else 0,
        "std_latency_ms": np.std(latencies) if latencies else 0,
        "deterministic": all_pass_count in [0, n_runs],  # All same outcome
    }


def run_ablation_experiment(
    dataset_name: str,
    cases: list[dict],
    model: str,
    n_runs: int = 5,
    limit: Optional[int] = None
) -> dict:
    """Run full ablation with multi-run on a dataset."""
    print(f"\n{'='*70}")
    print(f"ABLATION EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Cases: {len(cases)} | Runs per condition: {n_runs} | Model: {model}")
    print(f"Conditions: A (baseline), B (+wrapper), C (+rules), D (full)")
    
    if limit:
        cases = cases[:limit]
    
    total_runs = len(cases) * 4 * n_runs  # 4 conditions
    print(f"Total LLM calls: {total_runs}")
    print(f"Estimated time: {total_runs * 3 / 60:.1f} minutes (@ 3s/call)")
    
    all_results = []
    conditions = ["A", "B", "C", "D"]
    
    for i, case in enumerate(cases):
        case_id = case.get("id", f"case-{i}")
        print(f"\n[{i+1}/{len(cases)}] {case_id}")
        
        for cond in conditions:
            print(f"  Condition {cond} (N={n_runs})...", end=" ", flush=True)
            
            runs = run_case_multirun(case, cond, model, n_runs)
            all_results.extend(runs)
            
            agg = aggregate_multirun_results(runs)
            print(f"{agg['all_pass_rate']:.0%} ± [{agg['ci_95_lower']:.2f}, {agg['ci_95_upper']:.2f}]")
    
    # Dataset-level aggregation
    dataset_agg = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r.condition == cond]
        
        # Group by case_id
        by_case = {}
        for r in cond_results:
            if r.case_id not in by_case:
                by_case[r.case_id] = []
            by_case[r.case_id].append(r)
        
        # Aggregate per case, then dataset-wide
        case_aggs = [aggregate_multirun_results(runs) for runs in by_case.values()]
        
        dataset_pass_rate = np.mean([agg["all_pass_rate"] for agg in case_aggs])
        dataset_deterministic_pct = np.mean([agg["deterministic"] for agg in case_aggs]) * 100
        
        dataset_agg[cond] = {
            "mean_pass_rate": dataset_pass_rate,
            "deterministic_pct": dataset_deterministic_pct,
            "case_aggregates": case_aggs
        }
    
    print(f"\n{'-'*70}")
    print(f"DATASET SUMMARY: {dataset_name}")
    print(f"{'-'*70}")
    for cond in conditions:
        agg = dataset_agg[cond]
        print(f"Condition {cond}: {agg['mean_pass_rate']:.1%} pass rate | {agg['deterministic_pct']:.1f}% deterministic")
    
    return {
        "dataset": dataset_name,
        "model": model,
        "n_runs": n_runs,
        "timestamp": datetime.now().isoformat(),
        "all_results": [asdict(r) for r in all_results],
        "dataset_aggregates": dataset_agg
    }


def generate_ablation_latex_table(results: dict, output_path: str):
    """Generate LaTeX table for ablation results."""
    dataset = results["dataset"]
    agg = results["dataset_aggregates"]
    
    latex = r"""\\begin{table}[h]
\\centering
\\caption{Ablation study results with 95\% Wilson CI (N=5 per case).}
\\label{tab:ablation-results}
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\textbf{Condition} & \\textbf{Pass Rate} & \\textbf{95\% CI} & \\textbf{Deterministic} \\\\
\\midrule
"""
    
    condition_labels = {
        "A": "Baseline",
        "B": "+ Wrapper",
        "C": "+ Rules",
        "D": "Full Improved"
    }
    
    for cond in ["A", "B", "C", "D"]:
        label = condition_labels[cond]
        pass_rate = agg[cond]["mean_pass_rate"] * 100
        det_pct = agg[cond]["deterministic_pct"]
        
        # Calculate average CI width (simplified)
        latex += f"{label} & {pass_rate:.1f}\\% & [varies] & {det_pct:.1f}\\% \\\\\\\\\n"
    
    latex += r"""\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation + multi-run experiment")
    parser.add_argument("--dataset", choices=["extraction", "rag", "instruction", "all"], default="all")
    parser.add_argument("--runs", type=int, default=5, help="Runs per case per condition")
    parser.add_argument("--limit", type=int, default=None, help="Limit cases (for testing)")
    parser.add_argument("--model", default=CANDIDATE_MODEL)
    
    args = parser.parse_args()
    
    # Check Ollama
    if not check_ollama_available():
        print("ERROR: Ollama not running. Start with 'ollama serve'")
        return
    
    print(f"✓ Ollama available | Model: {args.model}")
    
    # Paths
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"
    results_dir = base_dir / "results_ablation"
    results_dir.mkdir(exist_ok=True)
    
    # Load datasets
    datasets = {}
    if args.dataset in ["extraction", "all"]:
        datasets["extraction"] = load_dataset(datasets_dir / "extraction_cases.jsonl")
    if args.dataset in ["rag", "all"]:
        datasets["rag"] = load_dataset(datasets_dir / "rag_cases.jsonl")
    if args.dataset in ["instruction", "all"]:
        datasets["instruction"] = load_dataset(datasets_dir / "instruction_cases.jsonl")
    
    # Run ablation experiments
    for name, cases in datasets.items():
        result = run_ablation_experiment(
            name, cases, args.model,
            n_runs=args.runs,
            limit=args.limit
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"{name}_ablation_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved: {result_path}")
        
        # Generate table
        latex_path = results_dir / f"{name}_ablation_table_{timestamp}.tex"
        generate_ablation_latex_table(result, latex_path)
    
    print("\n" + "="*70)
    print("ABLATION EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
