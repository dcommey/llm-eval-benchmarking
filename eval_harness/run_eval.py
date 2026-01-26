"""
Comprehensive evaluation runner with baseline vs improved prompt comparison.
Follows Sabit Ekin's iterative improvement methodology.
"""

import os
import sys
import json
import re
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CANDIDATE_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from ollama_runner import generate, check_ollama_available, list_models
from assertions import is_valid_json, extract_json_from_text, parse_json, has_required_fields, contains_any
from llm_judge import run_position_bias_test
from metrics import PositionBiasMetrics


# =============================================================================
# Baseline vs Improved Prompts (Ekin-style before/after)
# =============================================================================

BASELINE_SYSTEM_PROMPT = "You are a helpful assistant."

IMPROVED_SYSTEM_PROMPT = """You are a precise assistant that follows instructions exactly.
CRITICAL RULES:
1. Output ONLY what is requested - no explanations, no preamble, no follow-up
2. For JSON output: return valid JSON only, no markdown code blocks
3. For citations: use [1], [2] format and ONLY cite provided sources
4. If information is not in provided sources, say "I don't know"
5. Follow all formatting constraints exactly (word count, bullet count, etc.)"""


# =============================================================================
# Check Functions
# =============================================================================

def check_json_valid(response: str) -> tuple[bool, str]:
    """Check if response is valid JSON."""
    extracted = extract_json_from_text(response)
    if extracted and is_valid_json(extracted):
        return True, "Valid JSON"
    if is_valid_json(response.strip()):
        return True, "Valid JSON (raw)"
    return False, "Invalid JSON"


def check_required_keys(response: str, keys: list[str]) -> tuple[bool, list[str]]:
    """Check if JSON has required keys."""
    extracted = extract_json_from_text(response) or response.strip()
    parsed = parse_json(extracted)
    if not isinstance(parsed, dict):
        return False, keys
    present, missing = has_required_fields(parsed, keys)
    return present, missing


def check_allowed_values(response: str, field: str, allowed: list) -> tuple[bool, str]:
    """Check if a field value is in allowed list."""
    extracted = extract_json_from_text(response) or response.strip()
    parsed = parse_json(extracted)
    if not isinstance(parsed, dict):
        return False, "Not a JSON object"
    value = parsed.get(field)
    if value in allowed:
        return True, f"{field}={value}"
    return False, f"{field}={value} not in {allowed}"


def check_citations(response: str) -> tuple[bool, list[str]]:
    """Check for citation markers [1], [2], etc."""
    citations = re.findall(r'\[\d+\]', response)
    return len(citations) > 0, citations


def check_contains_all(response: str, terms: list[str]) -> tuple[bool, list[str]]:
    """Check if response contains all required terms."""
    response_lower = response.lower()
    missing = [t for t in terms if t.lower() not in response_lower]
    return len(missing) == 0, missing


def check_bullet_count(response: str, expected: int) -> tuple[bool, int]:
    """Count bullet points in response."""
    bullets = re.findall(r'^[\s]*[-•*]\s', response, re.MULTILINE)
    return len(bullets) == expected, len(bullets)


def check_sentence_count(response: str, expected: int) -> tuple[bool, int]:
    """Count sentences in response."""
    sentences = re.split(r'[.!?]+', response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) == expected, len(sentences)


def check_word_count(response: str, max_words: int = None, exact: int = None) -> tuple[bool, int]:
    """Check word count constraints."""
    words = len(response.split())
    if exact:
        return words == exact, words
    if max_words:
        return words <= max_words, words
    return True, words


def check_regex(response: str, pattern: str) -> tuple[bool, str]:
    """Check if response matches regex pattern."""
    try:
        if re.match(pattern, response.strip()):
            return True, "Matches"
        return False, f"Does not match {pattern}"
    except re.error as e:
        return False, f"Invalid regex: {e}"


def check_refusal(response: str) -> tuple[bool, str]:
    """Check if response is a refusal."""
    refusal_phrases = [
        "i can't", "i cannot", "i'm not able", "i am not able",
        "i won't", "i will not", "i'm sorry", "i apologize",
        "not able to", "cannot help", "can't assist", "cannot assist",
        "against my", "not appropriate", "not something i can"
    ]
    response_lower = response.lower()
    for phrase in refusal_phrases:
        if phrase in response_lower:
            return True, f"Refused with: {phrase}"
    return False, "No refusal detected"


def check_must_equal(response: str, expected: str) -> tuple[bool, str]:
    """Check if response exactly equals expected (case-insensitive, stripped)."""
    if response.strip().lower() == expected.lower():
        return True, "Exact match"
    return False, f"Expected '{expected}', got '{response.strip()[:50]}'"


# =============================================================================
# Evaluation Runner
# =============================================================================

@dataclass
class CaseResult:
    case_id: str
    task_type: str
    category: str
    prompt_version: str  # "baseline" or "improved"
    success: bool
    latency_ms: float
    checks_passed: int
    checks_total: int
    check_details: list
    response_preview: str


def run_checks(response: str, checks: dict, case: dict) -> tuple[int, int, list]:
    """Run all checks for a case and return (passed, total, details)."""
    details = []
    passed = 0
    total = 0
    
    # JSON validity
    if checks.get("json_valid"):
        total += 1
        ok, msg = check_json_valid(response)
        details.append({"check": "json_valid", "passed": ok, "message": msg})
        if ok:
            passed += 1
    
    # Required keys
    if checks.get("required_keys"):
        total += 1
        keys = checks["required_keys"]
        ok, missing = check_required_keys(response, keys)
        details.append({"check": "required_keys", "passed": ok, "missing": missing})
        if ok:
            passed += 1
    
    # Allowed values
    if checks.get("allowed_values"):
        for field, allowed in checks["allowed_values"].items():
            total += 1
            ok, msg = check_allowed_values(response, field, allowed)
            details.append({"check": f"allowed_values.{field}", "passed": ok, "message": msg})
            if ok:
                passed += 1
    
    # Must cite
    if checks.get("must_cite"):
        total += 1
        ok, citations = check_citations(response)
        details.append({"check": "must_cite", "passed": ok, "citations": citations})
        if ok:
            passed += 1
    
    # Must include terms
    if checks.get("must_include"):
        includes = checks["must_include"]
        if isinstance(includes, list):
            total += 1
            ok, missing = check_contains_all(response, includes)
            details.append({"check": "must_include", "passed": ok, "missing": missing})
            if ok:
                passed += 1
        elif isinstance(includes, dict):
            for field, terms in includes.items():
                total += 1
                extracted = extract_json_from_text(response) or response
                parsed = parse_json(extracted)
                field_value = parsed.get(field, "") if isinstance(parsed, dict) else ""
                ok, missing = check_contains_all(str(field_value), terms)
                details.append({"check": f"must_include.{field}", "passed": ok, "missing": missing})
                if ok:
                    passed += 1
    
    # Bullet count
    if checks.get("bullet_count"):
        total += 1
        ok, count = check_bullet_count(response, checks["bullet_count"])
        details.append({"check": "bullet_count", "passed": ok, "expected": checks["bullet_count"], "actual": count})
        if ok:
            passed += 1
    
    # Sentence count
    if checks.get("sentence_count"):
        total += 1
        ok, count = check_sentence_count(response, checks["sentence_count"])
        details.append({"check": "sentence_count", "passed": ok, "expected": checks["sentence_count"], "actual": count})
        if ok:
            passed += 1
    
    # Word count
    if checks.get("max_words"):
        total += 1
        ok, count = check_word_count(response, max_words=checks["max_words"])
        details.append({"check": "max_words", "passed": ok, "max": checks["max_words"], "actual": count})
        if ok:
            passed += 1
    
    if checks.get("answer_word_count"):
        total += 1
        extracted = extract_json_from_text(response) or response
        parsed = parse_json(extracted)
        answer = parsed.get("answer", "") if isinstance(parsed, dict) else response
        ok, count = check_word_count(answer, exact=checks["answer_word_count"])
        details.append({"check": "answer_word_count", "passed": ok, "expected": checks["answer_word_count"], "actual": count})
        if ok:
            passed += 1
    
    # Regex match
    if checks.get("regex"):
        total += 1
        ok, msg = check_regex(response, checks["regex"])
        details.append({"check": "regex", "passed": ok, "message": msg})
        if ok:
            passed += 1
    
    # Refusal
    if checks.get("must_refuse"):
        total += 1
        ok, msg = check_refusal(response)
        details.append({"check": "must_refuse", "passed": ok, "message": msg})
        if ok:
            passed += 1
    
    # Exact match
    if checks.get("must_equal"):
        total += 1
        ok, msg = check_must_equal(response, checks["must_equal"])
        details.append({"check": "must_equal", "passed": ok, "message": msg})
        if ok:
            passed += 1
    
    # Single sentence
    if checks.get("single_sentence"):
        total += 1
        ok, count = check_sentence_count(response, 1)
        details.append({"check": "single_sentence", "passed": ok, "sentence_count": count})
        if ok:
            passed += 1
    
    # Must end with period
    if checks.get("must_end_with_period"):
        total += 1
        ok = response.strip().endswith(".")
        details.append({"check": "must_end_with_period", "passed": ok})
        if ok:
            passed += 1
    
    return passed, total, details


def build_prompt(case: dict, use_improved: bool = False) -> str:
    """Build the full prompt for a case."""
    task_type = case.get("task_type", "")
    
    if task_type == "extraction":
        return case.get("prompt", "")
    
    elif task_type == "rag_qa":
        # Build RAG prompt with sources
        question = case.get("question", "")
        sources = case.get("sources", [])
        base_prompt = case.get("prompt", "Answer using ONLY the sources. Cite like [1].")
        
        sources_text = "\n".join([f"[{s['id']}] {s['text']}" for s in sources])
        full_prompt = f"SOURCES:\n{sources_text}\n\nQUESTION: {question}\n\nINSTRUCTIONS: {base_prompt}"
        return full_prompt
    
    elif task_type == "instruction_following":
        return case.get("prompt", "")
    
    return case.get("prompt", str(case))


def evaluate_case(
    case: dict,
    model: str,
    use_improved: bool = False,
    dry_run: bool = False
) -> CaseResult:
    """Evaluate a single case with baseline or improved prompt."""
    case_id = case.get("id", "unknown")
    task_type = case.get("task_type", "unknown")
    category = case.get("category", "unknown")
    checks = case.get("checks", {})
    
    prompt = build_prompt(case, use_improved)
    system_prompt = IMPROVED_SYSTEM_PROMPT if use_improved else BASELINE_SYSTEM_PROMPT
    
    if dry_run:
        # Mock response for dry run
        from ollama_runner import LLMResponse
        if task_type == "extraction":
            response = LLMResponse(
                text='{"name": "Test", "email": "test@test.com", "phone": null, "company": null}',
                model=model, latency_ms=100, prompt_tokens=50, completion_tokens=30, success=True
            )
        elif task_type == "rag_qa":
            response = LLMResponse(
                text='The warranty is 2 years [1].',
                model=model, latency_ms=120, prompt_tokens=100, completion_tokens=20, success=True
            )
        else:
            response = LLMResponse(
                text='TASK-1234',
                model=model, latency_ms=80, prompt_tokens=40, completion_tokens=10, success=True
            )
    else:
        response = generate(
            prompt,
            model,
            system_prompt=system_prompt
        )
    
    if not response.success:
        return CaseResult(
            case_id=case_id,
            task_type=task_type,
            category=category,
            prompt_version="improved" if use_improved else "baseline",
            success=False,
            latency_ms=response.latency_ms,
            checks_passed=0,
            checks_total=1,
            check_details=[{"error": response.error}],
            response_preview=""
        )
    
    passed, total, details = run_checks(response.text, checks, case)
    
    return CaseResult(
        case_id=case_id,
        task_type=task_type,
        category=category,
        prompt_version="improved" if use_improved else "baseline",
        success=response.success,
        latency_ms=response.latency_ms,
        checks_passed=passed,
        checks_total=total,
        check_details=details,
        response_preview=response.text[:200]
    )


def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset."""
    cases = []
    if not os.path.exists(path):
        print(f"Warning: Dataset not found at {path}")
        return []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_position_bias_experiment(
    cases: list[dict],
    judge_model: str,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict:
    """Run Position Bias experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT: POSITION BIAS")
    print(f"{'='*60}")
    
    if limit:
        cases = cases[:limit]

    print(f"Cases: {len(cases)} | Judge: {judge_model}")

    results = []
    bias_metrics = PositionBiasMetrics()

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case.get('id', 'unknown')}")
        
        question = case.get("question", "")
        response_a = case.get("response_a", "")
        response_b = case.get("response_b", "")

        print("  Running bias test...", end=" ", flush=True)
        if dry_run:
            bias_result = {
                "verdict_ab": {"winner": "A", "success": True},
                "verdict_ba": {"winner": "A", "success": True}, # Inconsistent
                "consistent": False,
                "first_position_preference_ab": True,
                "first_position_preference_ba": False
            }
        else:
            bias_result = run_position_bias_test(question, response_a, response_b, judge_model)
        
        # Update metrics
        bias_metrics.total_pairs += 1
        if not bias_result["consistent"]:
            bias_metrics.flips += 1
        if bias_result["verdict_ab"]["winner"] == "A":
            bias_metrics.first_position_wins += 1
        elif bias_result["verdict_ab"]["winner"] == "B":
            bias_metrics.second_position_wins += 1
        else:
            bias_metrics.ties += 1

        status = "✓" if bias_result["consistent"] else "⚠ FLIP"
        print(f"{status}")
        results.append({
            "case_id": case.get("id"),
            **bias_result
        })

    print(f"\n{'-'*60}")
    print("RESULTS: POSITION BIAS")
    print(f"{'-'*60}")
    print(f"Flips: {bias_metrics.flips}/{bias_metrics.total_pairs} ({bias_metrics.flip_rate():.1%})")
    print(f"Pos Bias: {bias_metrics.position_bias():+.1%}")

    return {
        "dataset": "position_bias",
        "model": judge_model,
        "timestamp": datetime.now().isoformat(),
        "metrics": bias_metrics.to_dict(),
        "results": results
    }



def run_evaluation(
    dataset_name: str,
    cases: list[dict],
    model: str,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict:
    """Run full evaluation on a dataset with baseline and improved prompts."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    if limit:
        cases = cases[:limit]
    
    print(f"Cases: {len(cases)} | Model: {model} | Dry run: {dry_run}")
    
    baseline_results = []
    improved_results = []
    
    for i, case in enumerate(cases):
        case_id = case.get("id", f"case-{i}")
        print(f"\n[{i+1}/{len(cases)}] {case_id}")
        
        # Run baseline
        print("  → Baseline...", end=" ", flush=True)
        baseline = evaluate_case(case, model, use_improved=False, dry_run=dry_run)
        baseline_results.append(baseline)
        status = "✓" if baseline.checks_passed == baseline.checks_total else f"✗ ({baseline.checks_passed}/{baseline.checks_total})"
        print(status)
        
        # Run improved
        print("  → Improved...", end=" ", flush=True)
        improved = evaluate_case(case, model, use_improved=True, dry_run=dry_run)
        improved_results.append(improved)
        status = "✓" if improved.checks_passed == improved.checks_total else f"✗ ({improved.checks_passed}/{improved.checks_total})"
        print(status)
    
    # Aggregate metrics
    def aggregate(results: list[CaseResult]) -> dict:
        total_checks = sum(r.checks_total for r in results)
        passed_checks = sum(r.checks_passed for r in results)
        all_passed = sum(1 for r in results if r.checks_passed == r.checks_total and r.checks_total > 0)
        latencies = [r.latency_ms for r in results if r.success]
        
        return {
            "total_cases": len(results),
            "all_checks_passed": all_passed,
            "all_pass_rate": round(all_passed / len(results), 4) if results else 0,
            "total_checks": total_checks,
            "checks_passed": passed_checks,
            "check_pass_rate": round(passed_checks / total_checks, 4) if total_checks else 0,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0
        }
    
    baseline_agg = aggregate(baseline_results)
    improved_agg = aggregate(improved_results)
    
    improvement = {
        "all_pass_rate_delta": round(improved_agg["all_pass_rate"] - baseline_agg["all_pass_rate"], 4),
        "check_pass_rate_delta": round(improved_agg["check_pass_rate"] - baseline_agg["check_pass_rate"], 4)
    }
    
    print(f"\n{'-'*60}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'-'*60}")
    print(f"Baseline: {baseline_agg['all_pass_rate']:.1%} all-pass | {baseline_agg['check_pass_rate']:.1%} check-pass")
    print(f"Improved: {improved_agg['all_pass_rate']:.1%} all-pass | {improved_agg['check_pass_rate']:.1%} check-pass")
    print(f"Δ: +{improvement['all_pass_rate_delta']*100:.1f}pp all-pass | +{improvement['check_pass_rate_delta']*100:.1f}pp check-pass")
    
    return {
        "dataset": dataset_name,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "aggregate": baseline_agg,
            "results": [asdict(r) for r in baseline_results]
        },
        "improved": {
            "aggregate": improved_agg,
            "results": [asdict(r) for r in improved_results]
        },
        "improvement": improvement
    }


def generate_latex_table(results: list[dict], output_path: str):
    """Generate LaTeX results table."""
    latex = r"""\begin{table}[h]
\centering
\caption{Evaluation results: Baseline vs. Improved prompts.}
\label{tab:eval-results}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Dataset} & \textbf{Cases} & \multicolumn{2}{c}{\textbf{Baseline}} & \multicolumn{2}{c}{\textbf{Improved}} & \textbf{$\Delta$} \\
 & & Pass\% & Check\% & Pass\% & Check\% & \\
\midrule
"""
    
    for r in results:
        ds = r["dataset"].replace("_", " ").title()
        n = r["baseline"]["aggregate"]["total_cases"]
        b_pass = r["baseline"]["aggregate"]["all_pass_rate"] * 100
        b_check = r["baseline"]["aggregate"]["check_pass_rate"] * 100
        i_pass = r["improved"]["aggregate"]["all_pass_rate"] * 100
        i_check = r["improved"]["aggregate"]["check_pass_rate"] * 100
        delta = r["improvement"]["check_pass_rate_delta"] * 100
        
        latex += f"{ds} & {n} & {b_pass:.1f} & {b_check:.1f} & {i_pass:.1f} & {i_check:.1f} & +{delta:.1f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table written to: {output_path}")


def generate_csv_summary(results: list[dict], output_path: str):
    """Generate CSV summary."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Cases", 
            "Baseline_AllPass%", "Baseline_CheckPass%", "Baseline_LatencyMs",
            "Improved_AllPass%", "Improved_CheckPass%", "Improved_LatencyMs",
            "Delta_AllPass", "Delta_CheckPass"
        ])
        
        for r in results:
            writer.writerow([
                r["dataset"],
                r["baseline"]["aggregate"]["total_cases"],
                round(r["baseline"]["aggregate"]["all_pass_rate"] * 100, 1),
                round(r["baseline"]["aggregate"]["check_pass_rate"] * 100, 1),
                r["baseline"]["aggregate"]["mean_latency_ms"],
                round(r["improved"]["aggregate"]["all_pass_rate"] * 100, 1),
                round(r["improved"]["aggregate"]["check_pass_rate"] * 100, 1),
                r["improved"]["aggregate"]["mean_latency_ms"],
                round(r["improvement"]["all_pass_rate_delta"] * 100, 1),
                round(r["improvement"]["check_pass_rate_delta"] * 100, 1)
            ])
    
    print(f"CSV summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive LLM evaluation")
    parser.add_argument("--dataset", choices=["extraction", "rag", "instruction", "bias", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock data")
    parser.add_argument("--limit", type=int, default=None, help="Limit cases per dataset")
    parser.add_argument("--model", default=CANDIDATE_MODEL, help="Model to evaluate")
    
    args = parser.parse_args()
    
    # Check Ollama
    if not args.dry_run:
        if not check_ollama_available():
            print("ERROR: Ollama is not running. Start with 'ollama serve' or use --dry-run")
            return
        print(f"✓ Ollama available | Model: {args.model}")
    
    # Paths
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load datasets
    datasets = {}
    if args.dataset in ["extraction", "all"]:
        datasets["extraction"] = load_dataset(datasets_dir / "extraction_cases.jsonl")
    if args.dataset in ["rag", "all"]:
        datasets["rag"] = load_dataset(datasets_dir / "rag_cases.jsonl")
    if args.dataset in ["instruction", "all"]:
        datasets["instruction"] = load_dataset(datasets_dir / "instruction_cases.jsonl")
    
    # Run evaluations
    all_results = []
    
    # Position Bias (Special handling)
    if args.dataset in ["bias", "all"]:
        # Only run if bias cases exist (they might be in instruction_cases or separate?)
        # For now let's assume a bias_cases.jsonl or use subset of instruction
        bias_path = datasets_dir / "bias_cases.jsonl"
        if bias_path.exists():
            bias_cases = load_dataset(bias_path)
            bias_res = run_position_bias_experiment(bias_cases, args.model, args.dry_run, args.limit)
            
            # Save bias result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = results_dir / f"position_bias_{timestamp}.json"
            with open(result_path, 'w') as f:
                json.dump(bias_res, f, indent=2)
            print(f"Results saved: {result_path}")
        else:
            if args.dataset == "bias":
                print("Warning: bias_cases.jsonl not found.")
        result = run_evaluation(
            name, cases, args.model,
            dry_run=args.dry_run,
            limit=args.limit
        )
        all_results.append(result)
        
        # Save individual result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"{name}_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved: {result_path}")
    
    # Generate aggregated outputs
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generate_csv_summary(all_results, results_dir / f"summary_{timestamp}.csv")
        generate_latex_table(all_results, results_dir / f"results_table_{timestamp}.tex")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
