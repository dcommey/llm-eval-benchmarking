"""
Main experiment runner for LLM evaluation experiments.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import CANDIDATE_MODEL, JUDGE_MODEL, TESTSET_PATH, RESULTS_DIR
from ollama_runner import generate, check_ollama_available, list_models, LLMResponse
from assertions import run_all_assertions, extract_json_from_text, is_valid_json, has_required_fields
from metrics import compute_json_metrics, aggregate_metrics, PositionBiasMetrics
from llm_judge import judge_pairwise, judge_rubric, run_position_bias_test


def load_testset(path: str) -> list[dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def save_results(results: dict, experiment_name: str, results_dir: str = RESULTS_DIR):
    """Save experiment results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


# =============================================================================
# Experiment 1: Structured Output Reliability
# =============================================================================

BASELINE_EXTRACTION_PROMPT = """Extract the following information from the text and return it as JSON:
- name
- date
- location
- amount

Text: {text}

Return only the JSON, no other text."""

IMPROVED_EXTRACTION_PROMPT = """Extract the following information from the text and return it as valid JSON.

Required fields (use null if not found):
- name (string): Person or organization name
- date (string): Any date mentioned, in YYYY-MM-DD format if possible
- location (string): Any place mentioned
- amount (number or null): Any monetary amount as a number

Text: {text}

Example output format:
{{"name": "John Smith", "date": "2024-03-15", "location": "New York", "amount": 1500.00}}

Return only the JSON object, no other text or explanation."""


def run_structured_output_experiment(
    test_cases: list[dict],
    model: str = CANDIDATE_MODEL,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict:
    """
    Run Experiment 1: Structured output reliability.
    
    Compares baseline vs improved prompts for JSON generation.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Structured Output Reliability")
    print("="*60)
    
    # Filter to extraction tasks
    extraction_cases = [c for c in test_cases if c.get("task_type") == "extraction"]
    if limit:
        extraction_cases = extraction_cases[:limit]
    
    print(f"Running on {len(extraction_cases)} extraction cases with model: {model}")
    
    results = {
        "experiment": "structured_output",
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "cases": [],
        "baseline": {"results": []},
        "improved": {"results": []}
    }
    
    required_fields = ["name", "date", "location", "amount"]
    
    for i, case in enumerate(extraction_cases):
        print(f"\n[{i+1}/{len(extraction_cases)}] Case: {case.get('id', 'unknown')}")
        
        text = case.get("input", "")
        
        # Run baseline prompt
        baseline_prompt = BASELINE_EXTRACTION_PROMPT.format(text=text)
        print("  Running baseline prompt...")
        
        if dry_run:
            baseline_response = LLMResponse(
                text='{"name": "Test", "date": "2024-01-01", "location": "Test City", "amount": 100}',
                model=model, latency_ms=100, prompt_tokens=50, completion_tokens=30, success=True
            )
        else:
            baseline_response = generate(baseline_prompt, model)
        
        baseline_metrics = compute_json_metrics(baseline_response.text, required_fields)
        results["baseline"]["results"].append({
            "case_id": case.get("id"),
            "success": baseline_response.success,
            "latency_ms": baseline_response.latency_ms,
            "json_valid": baseline_metrics["is_valid"],
            "has_required_fields": baseline_metrics["has_required_fields"],
            "missing_fields": baseline_metrics["missing_fields"],
            "response": baseline_response.text[:500]  # Truncate for storage
        })
        
        # Run improved prompt
        improved_prompt = IMPROVED_EXTRACTION_PROMPT.format(text=text)
        print("  Running improved prompt...")
        
        if dry_run:
            improved_response = LLMResponse(
                text='{"name": "Test", "date": "2024-01-01", "location": "Test City", "amount": 100}',
                model=model, latency_ms=95, prompt_tokens=80, completion_tokens=30, success=True
            )
        else:
            improved_response = generate(improved_prompt, model)
        
        improved_metrics = compute_json_metrics(improved_response.text, required_fields)
        results["improved"]["results"].append({
            "case_id": case.get("id"),
            "success": improved_response.success,
            "latency_ms": improved_response.latency_ms,
            "json_valid": improved_metrics["is_valid"],
            "has_required_fields": improved_metrics["has_required_fields"],
            "missing_fields": improved_metrics["missing_fields"],
            "response": improved_response.text[:500]
        })
        
        status = "✓" if improved_metrics["is_valid"] else "✗"
        print(f"  {status} Baseline JSON valid: {baseline_metrics['is_valid']}, Improved: {improved_metrics['is_valid']}")
    
    # Aggregate metrics
    results["baseline"]["aggregate"] = aggregate_metrics(results["baseline"]["results"])
    results["improved"]["aggregate"] = aggregate_metrics(results["improved"]["results"])
    
    # Compute improvement
    baseline_rate = results["baseline"]["aggregate"]["json_valid_rate"]
    improved_rate = results["improved"]["aggregate"]["json_valid_rate"]
    results["improvement"] = {
        "json_valid_rate_delta": round(improved_rate - baseline_rate, 4),
        "json_valid_rate_delta_pct": round((improved_rate - baseline_rate) * 100, 2)
    }
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"Baseline JSON valid rate: {baseline_rate:.1%}")
    print(f"Improved JSON valid rate: {improved_rate:.1%}")
    print(f"Improvement: +{results['improvement']['json_valid_rate_delta_pct']:.1f}%")
    
    return results


# =============================================================================
# Experiment 2: Prompt Engineering Impact
# =============================================================================

V1_INSTRUCTION_PROMPT = """{instruction}

{input}"""

V2_INSTRUCTION_PROMPT = """You are a helpful assistant. Follow the instructions carefully.

INSTRUCTIONS:
{instruction}

INPUT:
{input}

REQUIREMENTS:
- Follow all formatting requirements exactly
- Be concise but complete
- If a specific format is requested, use that format

OUTPUT:"""


def run_prompt_iteration_experiment(
    test_cases: list[dict],
    model: str = CANDIDATE_MODEL,
    judge_model: str = JUDGE_MODEL,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict:
    """
    Run Experiment 2: Prompt engineering impact.
    
    Compares minimal vs detailed prompts with rubric scoring.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Prompt Engineering Impact")
    print("="*60)
    
    # Filter to instruction-following tasks
    instruction_cases = [c for c in test_cases if c.get("task_type") == "instruction_following"]
    if limit:
        instruction_cases = instruction_cases[:limit]
    
    print(f"Running on {len(instruction_cases)} cases with model: {model}")
    
    results = {
        "experiment": "prompt_iteration",
        "model": model,
        "judge_model": judge_model,
        "timestamp": datetime.now().isoformat(),
        "v1": {"results": []},
        "v2": {"results": []}
    }
    
    for i, case in enumerate(instruction_cases):
        print(f"\n[{i+1}/{len(instruction_cases)}] Case: {case.get('id', 'unknown')}")
        
        instruction = case.get("instruction", "")
        input_text = case.get("input", "")
        assertions = case.get("assertions", [])
        
        # Run V1 (minimal) prompt
        v1_prompt = V1_INSTRUCTION_PROMPT.format(instruction=instruction, input=input_text)
        print("  Running V1 (minimal) prompt...")
        
        if dry_run:
            v1_response = LLMResponse(
                text="Here is the answer following the format.",
                model=model, latency_ms=150, prompt_tokens=50, completion_tokens=20, success=True
            )
        else:
            v1_response = generate(v1_prompt, model)
        
        # Check format compliance
        v1_assertions = run_all_assertions(assertions, v1_response.text) if assertions else {"pass_rate": 1.0}
        
        # Get rubric score from judge
        if dry_run:
            v1_judge = {"overall_score": 3.5, "success": True}
        else:
            v1_judge = judge_rubric(instruction, v1_response.text, judge_model=judge_model)
        
        results["v1"]["results"].append({
            "case_id": case.get("id"),
            "success": v1_response.success,
            "latency_ms": v1_response.latency_ms,
            "format_compliant": v1_assertions.get("pass_rate", 0) == 1.0,
            "assertion_pass_rate": v1_assertions.get("pass_rate", 0),
            "rubric_score": v1_judge.get("overall_score", 0),
            "response": v1_response.text[:500]
        })
        
        # Run V2 (detailed) prompt
        v2_prompt = V2_INSTRUCTION_PROMPT.format(instruction=instruction, input=input_text)
        print("  Running V2 (detailed) prompt...")
        
        if dry_run:
            v2_response = LLMResponse(
                text="Here is the answer following the exact format requested.",
                model=model, latency_ms=180, prompt_tokens=100, completion_tokens=25, success=True
            )
        else:
            v2_response = generate(v2_prompt, model)
        
        v2_assertions = run_all_assertions(assertions, v2_response.text) if assertions else {"pass_rate": 1.0}
        
        if dry_run:
            v2_judge = {"overall_score": 4.2, "success": True}
        else:
            v2_judge = judge_rubric(instruction, v2_response.text, judge_model=judge_model)
        
        results["v2"]["results"].append({
            "case_id": case.get("id"),
            "success": v2_response.success,
            "latency_ms": v2_response.latency_ms,
            "format_compliant": v2_assertions.get("pass_rate", 0) == 1.0,
            "assertion_pass_rate": v2_assertions.get("pass_rate", 0),
            "rubric_score": v2_judge.get("overall_score", 0),
            "response": v2_response.text[:500]
        })
        
        print(f"  V1 score: {v1_judge.get('overall_score', 0):.1f}, V2 score: {v2_judge.get('overall_score', 0):.1f}")
    
    # Aggregate metrics
    results["v1"]["aggregate"] = aggregate_metrics(results["v1"]["results"])
    results["v2"]["aggregate"] = aggregate_metrics(results["v2"]["results"])
    
    # Compute improvement
    v1_score = results["v1"]["aggregate"]["mean_rubric_score"]
    v2_score = results["v2"]["aggregate"]["mean_rubric_score"]
    v1_format = results["v1"]["aggregate"]["format_compliance_rate"]
    v2_format = results["v2"]["aggregate"]["format_compliance_rate"]
    
    results["improvement"] = {
        "rubric_score_delta": round(v2_score - v1_score, 2),
        "format_compliance_delta": round(v2_format - v1_format, 4),
        "format_compliance_delta_pct": round((v2_format - v1_format) * 100, 2)
    }
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"V1 mean rubric score: {v1_score:.2f}, format compliance: {v1_format:.1%}")
    print(f"V2 mean rubric score: {v2_score:.2f}, format compliance: {v2_format:.1%}")
    print(f"Improvement: +{results['improvement']['rubric_score_delta']:.2f} rubric, +{results['improvement']['format_compliance_delta_pct']:.1f}% format")
    
    return results


# =============================================================================
# Experiment 3: Position Bias Demonstration
# =============================================================================

def run_position_bias_experiment(
    test_cases: list[dict],
    judge_model: str = JUDGE_MODEL,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict:
    """
    Run Experiment 3: LLM-as-judge position bias demonstration.
    
    Tests each pair in both orders to measure position bias.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Position Bias Demonstration")
    print("="*60)
    
    # Filter to pairwise comparison tasks
    pairwise_cases = [c for c in test_cases if c.get("task_type") == "pairwise"]
    if limit:
        pairwise_cases = pairwise_cases[:limit]
    
    print(f"Running on {len(pairwise_cases)} pairs with judge model: {judge_model}")
    
    results = {
        "experiment": "position_bias",
        "judge_model": judge_model,
        "timestamp": datetime.now().isoformat(),
        "cases": [],
        "metrics": None
    }
    
    bias_metrics = PositionBiasMetrics()
    
    for i, case in enumerate(pairwise_cases):
        print(f"\n[{i+1}/{len(pairwise_cases)}] Case: {case.get('id', 'unknown')}")
        
        question = case.get("question", "")
        response_a = case.get("response_a", "")
        response_b = case.get("response_b", "")
        
        print("  Running position bias test (both orderings)...")
        
        if dry_run:
            # Simulate some position bias
            import random
            first_wins = random.random() < 0.55  # 55% first-position preference
            bias_result = {
                "verdict_ab": {"winner": "A" if first_wins else "B", "success": True},
                "verdict_ba": {"winner": "B" if first_wins else "A", "success": True},
                "consistent": not first_wins,  # Sometimes inconsistent
                "first_position_preference_ab": first_wins,
                "first_position_preference_ba": first_wins,
                "both_favor_first": first_wins
            }
        else:
            bias_result = run_position_bias_test(question, response_a, response_b, judge_model)
        
        results["cases"].append({
            "case_id": case.get("id"),
            **bias_result
        })
        
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
        
        status = "✓ consistent" if bias_result["consistent"] else "⚠ FLIP"
        print(f"  {status} | AB→{bias_result['verdict_ab']['winner']}, BA→{bias_result['verdict_ba']['winner']}")
    
    results["metrics"] = bias_metrics.to_dict()
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"Total pairs: {bias_metrics.total_pairs}")
    print(f"Preference flips: {bias_metrics.flips} ({bias_metrics.flip_rate():.1%})")
    print(f"First position preference: {bias_metrics.first_position_preference():.1%}")
    print(f"Position bias: {bias_metrics.position_bias():+.1%}")
    
    return results


# =============================================================================
# Main Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--structured", action="store_true", help="Run structured output experiment")
    parser.add_argument("--prompt", action="store_true", help="Run prompt iteration experiment")
    parser.add_argument("--bias", action="store_true", help="Run position bias experiment")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock data (no Ollama)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases per experiment")
    parser.add_argument("--testset", type=str, default=TESTSET_PATH, help="Path to test set JSONL")
    
    args = parser.parse_args()
    
    # Check Ollama availability
    if not args.dry_run:
        if not check_ollama_available():
            print("ERROR: Ollama is not running. Start Ollama first or use --dry-run.")
            return
        
        available_models = list_models()
        print(f"Available Ollama models: {available_models}")
        
        if CANDIDATE_MODEL not in available_models:
            print(f"WARNING: Candidate model '{CANDIDATE_MODEL}' not found. Run: ollama pull {CANDIDATE_MODEL}")
        if JUDGE_MODEL not in available_models:
            print(f"WARNING: Judge model '{JUDGE_MODEL}' not found. Run: ollama pull {JUDGE_MODEL}")
    
    # Load test set
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"ERROR: Test set not found at {testset_path}")
        print("Generate the test set first or specify --testset path.")
        return
    
    print(f"Loading test set from: {testset_path}")
    test_cases = load_testset(str(testset_path))
    print(f"Loaded {len(test_cases)} test cases")
    
    # Run selected experiments
    if args.all or args.structured:
        results = run_structured_output_experiment(
            test_cases, dry_run=args.dry_run, limit=args.limit
        )
        save_results(results, "structured_output")
    
    if args.all or args.prompt:
        results = run_prompt_iteration_experiment(
            test_cases, dry_run=args.dry_run, limit=args.limit
        )
        save_results(results, "prompt_iteration")
    
    if args.all or args.bias:
        results = run_position_bias_experiment(
            test_cases, dry_run=args.dry_run, limit=args.limit
        )
        save_results(results, "position_bias")
    
    if not any([args.all, args.structured, args.prompt, args.bias]):
        print("\nNo experiment selected. Use --all or specify experiments:")
        print("  --structured  : Structured output reliability")
        print("  --prompt      : Prompt engineering impact")
        print("  --bias        : Position bias demonstration")
        print("\nUse --dry-run for testing without Ollama")


if __name__ == "__main__":
    main()
