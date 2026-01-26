"""
Metrics computation for LLM evaluation experiments.
"""

import json
import statistics
from typing import Any
from dataclasses import dataclass, field

from assertions import is_valid_json, extract_json_from_text, has_required_fields


@dataclass
class ExperimentMetrics:
    """Container for experiment-level metrics."""
    total_cases: int = 0
    successful_calls: int = 0
    
    # Structured output metrics
    json_valid_count: int = 0
    required_fields_count: int = 0
    
    # Format compliance metrics
    format_pass_count: int = 0
    
    # Latency metrics
    latencies_ms: list[float] = field(default_factory=list)
    
    # Judge scores
    rubric_scores: list[float] = field(default_factory=list)
    
    def json_valid_rate(self) -> float:
        return self.json_valid_count / self.total_cases if self.total_cases > 0 else 0.0
    
    def required_fields_rate(self) -> float:
        return self.required_fields_count / self.total_cases if self.total_cases > 0 else 0.0
    
    def format_compliance_rate(self) -> float:
        return self.format_pass_count / self.total_cases if self.total_cases > 0 else 0.0
    
    def mean_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    def median_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0
    
    def std_latency_ms(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0
    
    def mean_rubric_score(self) -> float:
        return statistics.mean(self.rubric_scores) if self.rubric_scores else 0.0
    
    def success_rate(self) -> float:
        return self.successful_calls / self.total_cases if self.total_cases > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "successful_calls": self.successful_calls,
            "success_rate": round(self.success_rate(), 4),
            "json_valid_rate": round(self.json_valid_rate(), 4),
            "required_fields_rate": round(self.required_fields_rate(), 4),
            "format_compliance_rate": round(self.format_compliance_rate(), 4),
            "mean_latency_ms": round(self.mean_latency_ms(), 2),
            "median_latency_ms": round(self.median_latency_ms(), 2),
            "std_latency_ms": round(self.std_latency_ms(), 2),
            "mean_rubric_score": round(self.mean_rubric_score(), 2)
        }


def compute_exact_match(response: str, expected: str, normalize: bool = True) -> bool:
    """
    Check if response exactly matches expected output.
    
    Args:
        response: The LLM response
        expected: The expected output
        normalize: If True, normalize whitespace and case
    """
    if normalize:
        response = " ".join(response.lower().split())
        expected = " ".join(expected.lower().split())
    return response == expected


def compute_json_metrics(response: str, required_fields: list[str] = None) -> dict:
    """
    Compute JSON-related metrics for a response.
    
    Returns:
        Dict with 'is_valid', 'has_required_fields', 'parsed_json'
    """
    result = {
        "is_valid": False,
        "has_required_fields": False,
        "missing_fields": [],
        "parsed_json": None
    }
    
    # Try to parse JSON
    extracted = extract_json_from_text(response)
    if extracted:
        try:
            parsed = json.loads(extracted)
            result["is_valid"] = True
            result["parsed_json"] = parsed
            
            # Check required fields if specified
            if required_fields and isinstance(parsed, dict):
                has_all, missing = has_required_fields(parsed, required_fields)
                result["has_required_fields"] = has_all
                result["missing_fields"] = missing
            elif not required_fields:
                result["has_required_fields"] = True
                
        except json.JSONDecodeError:
            pass
    
    return result


def compute_format_compliance(response: str, assertions: list[dict]) -> dict:
    """
    Compute format compliance based on assertions.
    
    Returns:
        Dict with 'compliant', 'pass_count', 'fail_count', 'details'
    """
    from assertions import run_all_assertions
    
    results = run_all_assertions(assertions, response)
    return {
        "compliant": results["failed"] == 0,
        "pass_count": results["passed"],
        "fail_count": results["failed"],
        "pass_rate": results["pass_rate"],
        "details": results["details"]
    }


@dataclass
class PositionBiasMetrics:
    """Metrics for position bias experiment."""
    total_pairs: int = 0
    flips: int = 0  # Number of times preference changed with order
    first_position_wins: int = 0  # Preferences for first position
    second_position_wins: int = 0
    ties: int = 0
    
    def flip_rate(self) -> float:
        """Rate at which preferences flip when order is swapped."""
        return self.flips / self.total_pairs if self.total_pairs > 0 else 0.0
    
    def position_bias(self) -> float:
        """Bias toward first position (positive) or second (negative)."""
        decided = self.first_position_wins + self.second_position_wins
        if decided == 0:
            return 0.0
        return (self.first_position_wins - self.second_position_wins) / decided
    
    def first_position_preference(self) -> float:
        """Percentage of non-tie decisions favoring first position."""
        decided = self.first_position_wins + self.second_position_wins
        if decided == 0:
            return 0.5
        return self.first_position_wins / decided
    
    def to_dict(self) -> dict:
        return {
            "total_pairs": self.total_pairs,
            "flips": self.flips,
            "flip_rate": round(self.flip_rate(), 4),
            "first_position_wins": self.first_position_wins,
            "second_position_wins": self.second_position_wins,
            "ties": self.ties,
            "position_bias": round(self.position_bias(), 4),
            "first_position_preference": round(self.first_position_preference(), 4)
        }


def aggregate_metrics(case_results: list[dict]) -> dict:
    """
    Aggregate metrics across multiple test cases.
    
    Args:
        case_results: List of per-case result dicts
        
    Returns:
        Aggregated metrics dict
    """
    metrics = ExperimentMetrics()
    metrics.total_cases = len(case_results)
    
    for result in case_results:
        if result.get("success", False):
            metrics.successful_calls += 1
        
        if result.get("json_valid", False):
            metrics.json_valid_count += 1
            
        if result.get("has_required_fields", False):
            metrics.required_fields_count += 1
            
        if result.get("format_compliant", False):
            metrics.format_pass_count += 1
        
        if "latency_ms" in result:
            metrics.latencies_ms.append(result["latency_ms"])
            
        if "rubric_score" in result:
            metrics.rubric_scores.append(result["rubric_score"])
    
    return metrics.to_dict()
