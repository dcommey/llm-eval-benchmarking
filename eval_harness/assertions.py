"""
Automated assertions for evaluating LLM outputs.
"""

import json
import re
from typing import Any


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def parse_json(text: str) -> dict | list | None:
    """Parse JSON from text, returning None if invalid."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_json_from_text(text: str) -> str | None:
    """
    Extract JSON from text that may contain other content.
    Looks for content between ```json and ``` or { and }.
    """
    # Try to find JSON in code blocks
    code_block_match = re.search(r'```json?\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Try to find raw JSON object
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        candidate = brace_match.group(0)
        if is_valid_json(candidate):
            return candidate
    
    # Try to find raw JSON array
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        candidate = bracket_match.group(0)
        if is_valid_json(candidate):
            return candidate
    
    return None


def has_required_fields(data: dict, required_fields: list[str]) -> tuple[bool, list[str]]:
    """
    Check if all required fields are present in a dict.
    
    Returns:
        Tuple of (all_present, missing_fields)
    """
    missing = [f for f in required_fields if f not in data]
    return len(missing) == 0, missing


def check_field_types(data: dict, schema: dict[str, type]) -> tuple[bool, dict]:
    """
    Check if fields have expected types.
    
    Args:
        data: Dict to validate
        schema: Dict mapping field names to expected types
        
    Returns:
        Tuple of (all_valid, {field: actual_type} for invalid fields)
    """
    invalid = {}
    for field, expected_type in schema.items():
        if field in data and not isinstance(data[field], expected_type):
            invalid[field] = type(data[field]).__name__
    return len(invalid) == 0, invalid


def contains(text: str, substring: str, case_sensitive: bool = False) -> bool:
    """Check if text contains a substring."""
    if case_sensitive:
        return substring in text
    return substring.lower() in text.lower()


def contains_any(text: str, substrings: list[str], case_sensitive: bool = False) -> bool:
    """Check if text contains any of the substrings."""
    return any(contains(text, s, case_sensitive) for s in substrings)


def contains_all(text: str, substrings: list[str], case_sensitive: bool = False) -> bool:
    """Check if text contains all of the substrings."""
    return all(contains(text, s, case_sensitive) for s in substrings)


def not_contains(text: str, substrings: list[str], case_sensitive: bool = False) -> bool:
    """Check if text does not contain any of the substrings."""
    return not any(contains(text, s, case_sensitive) for s in substrings)


def matches_regex(text: str, pattern: str) -> bool:
    """Check if text matches a regex pattern."""
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return False


def length_in_range(text: str, min_length: int = 0, max_length: int = float('inf')) -> bool:
    """Check if text length is within range."""
    return min_length <= len(text) <= max_length


def word_count_in_range(text: str, min_words: int = 0, max_words: int = float('inf')) -> bool:
    """Check if word count is within range."""
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def check_citation_format(text: str) -> tuple[bool, list[str]]:
    """
    Check for citations in [1], [2] format.
    
    Returns:
        Tuple of (has_citations, list of citation markers found)
    """
    citations = re.findall(r'\[\d+\]', text)
    return len(citations) > 0, citations


def run_assertion(assertion: dict, text: str, parsed_json: Any = None) -> tuple[bool, str]:
    """
    Run a single assertion against text.
    
    Args:
        assertion: Dict with 'type' and parameters
        text: The text to check
        parsed_json: Pre-parsed JSON if available
        
    Returns:
        Tuple of (passed, reason)
    """
    assertion_type = assertion.get("type")
    
    if assertion_type == "valid_json":
        if is_valid_json(text):
            return True, "Valid JSON"
        # Try to extract JSON
        extracted = extract_json_from_text(text)
        if extracted and is_valid_json(extracted):
            return True, "Valid JSON (extracted from text)"
        return False, "Invalid JSON"
    
    elif assertion_type == "contains":
        value = assertion.get("value", "")
        if contains(text, value):
            return True, f"Contains '{value}'"
        return False, f"Does not contain '{value}'"
    
    elif assertion_type == "contains_any":
        values = assertion.get("values", [])
        if contains_any(text, values):
            return True, f"Contains one of {values}"
        return False, f"Does not contain any of {values}"
    
    elif assertion_type == "contains_all":
        values = assertion.get("values", [])
        if contains_all(text, values):
            return True, f"Contains all of {values}"
        missing = [v for v in values if not contains(text, v)]
        return False, f"Missing: {missing}"
    
    elif assertion_type == "not_contains":
        values = assertion.get("values", [])
        if not_contains(text, values):
            return True, f"Does not contain any of {values}"
        found = [v for v in values if contains(text, v)]
        return False, f"Contains prohibited: {found}"
    
    elif assertion_type == "matches_regex":
        pattern = assertion.get("pattern", "")
        if matches_regex(text, pattern):
            return True, f"Matches pattern '{pattern}'"
        return False, f"Does not match pattern '{pattern}'"
    
    elif assertion_type == "max_length_chars":
        max_len = assertion.get("value", float('inf'))
        if len(text) <= max_len:
            return True, f"Length {len(text)} <= {max_len}"
        return False, f"Length {len(text)} > {max_len}"
    
    elif assertion_type == "min_length_chars":
        min_len = assertion.get("value", 0)
        if len(text) >= min_len:
            return True, f"Length {len(text)} >= {min_len}"
        return False, f"Length {len(text)} < {min_len}"
    
    elif assertion_type == "has_required_fields":
        if parsed_json is None:
            extracted = extract_json_from_text(text)
            parsed_json = parse_json(extracted) if extracted else None
        if not isinstance(parsed_json, dict):
            return False, "Not a JSON object"
        fields = assertion.get("fields", [])
        has_all, missing = has_required_fields(parsed_json, fields)
        if has_all:
            return True, f"Has all required fields: {fields}"
        return False, f"Missing fields: {missing}"
    
    elif assertion_type == "has_citations":
        has_cites, cites = check_citation_format(text)
        if has_cites:
            return True, f"Has citations: {cites}"
        return False, "No citations found"
    
    else:
        return False, f"Unknown assertion type: {assertion_type}"


def run_all_assertions(assertions: list[dict], text: str) -> dict:
    """
    Run all assertions and return detailed results.
    
    Returns:
        Dict with 'passed', 'failed', 'total', and 'details'
    """
    # Pre-parse JSON if needed
    extracted = extract_json_from_text(text)
    parsed_json = parse_json(extracted) if extracted else parse_json(text)
    
    results = {
        "passed": 0,
        "failed": 0,
        "total": len(assertions),
        "details": []
    }
    
    for assertion in assertions:
        passed, reason = run_assertion(assertion, text, parsed_json)
        results["details"].append({
            "type": assertion.get("type"),
            "passed": passed,
            "reason": reason
        })
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
    return results
