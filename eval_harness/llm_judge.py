"""
LLM-as-Judge implementation with position bias mitigation.
"""

import json
import random
from typing import Optional
from dataclasses import dataclass

from ollama_runner import generate, LLMResponse
from config import JUDGE_MODEL


@dataclass
class JudgeVerdict:
    """Container for judge evaluation result."""
    winner: str  # "A", "B", or "tie"
    scores: dict[str, dict]  # {"A": {...}, "B": {...}}
    reasoning: str
    raw_response: str
    order: str  # "original" or "swapped"
    success: bool
    error: Optional[str] = None


PAIRWISE_JUDGE_PROMPT = """You are an expert evaluator. Compare the two responses below for the given question.

[Question]
{question}

[Response A]
{response_a}

[Response B]
{response_b}

Evaluate each response on these criteria (score 1-5 for each):
1. Accuracy: Are all facts correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-organized and easy to understand?
4. Conciseness: Is it appropriately brief without unnecessary filler?

First, analyze each response step by step.
Then provide your evaluation in this exact JSON format:
{{
  "analysis_a": "Brief analysis of Response A",
  "analysis_b": "Brief analysis of Response B",
  "scores_a": {{"accuracy": X, "completeness": X, "clarity": X, "conciseness": X}},
  "scores_b": {{"accuracy": X, "completeness": X, "clarity": X, "conciseness": X}},
  "winner": "A" or "B" or "tie",
  "reasoning": "Brief explanation of why you chose this winner"
}}

Respond only with the JSON, no other text."""


RUBRIC_JUDGE_PROMPT = """You are an expert evaluator. Score the following response on a 1-5 scale for each criterion.

[Question/Task]
{question}

[Response]
{response}

[Scoring Rubric]
{rubric}

Provide your evaluation in this exact JSON format:
{{
  "scores": {{
    "criterion_1": X,
    "criterion_2": X,
    ...
  }},
  "overall_score": X,
  "reasoning": "Brief explanation of scores"
}}

Respond only with the JSON, no other text."""


DEFAULT_RUBRIC = """
- Accuracy (1-5): Are all facts correct? 1=multiple errors, 5=perfectly accurate
- Completeness (1-5): Does it fully answer the question? 1=missing key info, 5=comprehensive
- Clarity (1-5): Is it well-organized? 1=confusing, 5=crystal clear
- Helpfulness (1-5): Would this help a user? 1=not useful, 5=very helpful
"""


def _parse_judge_json(text: str) -> dict | None:
    """Parse JSON from judge response, handling potential formatting issues."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from code blocks
    import re
    match = re.search(r'```json?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def judge_pairwise(
    question: str,
    response_a: str,
    response_b: str,
    randomize_order: bool = True,
    judge_model: str = JUDGE_MODEL
) -> JudgeVerdict:
    """
    Evaluate two responses using an LLM judge.
    
    Args:
        question: The original question/prompt
        response_a: First response to compare
        response_b: Second response to compare
        randomize_order: If True, randomly swap order to mitigate position bias
        judge_model: Model to use as judge
        
    Returns:
        JudgeVerdict with winner and scores
    """
    # Randomize order for position bias mitigation
    if randomize_order and random.random() < 0.5:
        first, second = response_b, response_a
        order = "swapped"
    else:
        first, second = response_a, response_b
        order = "original"
    
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        question=question,
        response_a=first,
        response_b=second
    )
    
    llm_response = generate(prompt, judge_model)
    
    if not llm_response.success:
        return JudgeVerdict(
            winner="error",
            scores={},
            reasoning="",
            raw_response="",
            order=order,
            success=False,
            error=llm_response.error
        )
    
    parsed = _parse_judge_json(llm_response.text)
    
    if not parsed:
        return JudgeVerdict(
            winner="error",
            scores={},
            reasoning="Failed to parse judge response",
            raw_response=llm_response.text,
            order=order,
            success=False,
            error="JSON parse error"
        )
    
    # Extract winner and correct for order swapping
    winner = parsed.get("winner", "tie")
    if order == "swapped":
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"
    
    # Also swap scores back
    scores = {}
    if order == "swapped":
        scores["A"] = parsed.get("scores_b", {})
        scores["B"] = parsed.get("scores_a", {})
    else:
        scores["A"] = parsed.get("scores_a", {})
        scores["B"] = parsed.get("scores_b", {})
    
    return JudgeVerdict(
        winner=winner,
        scores=scores,
        reasoning=parsed.get("reasoning", ""),
        raw_response=llm_response.text,
        order=order,
        success=True
    )


def judge_rubric(
    question: str,
    response: str,
    rubric: str = DEFAULT_RUBRIC,
    judge_model: str = JUDGE_MODEL
) -> dict:
    """
    Score a response using a rubric-based prompt.
    
    Args:
        question: The original question/prompt
        response: The response to evaluate
        rubric: Scoring rubric text
        judge_model: Model to use as judge
        
    Returns:
        Dict with scores, overall_score, reasoning, and success
    """
    prompt = RUBRIC_JUDGE_PROMPT.format(
        question=question,
        response=response,
        rubric=rubric
    )
    
    llm_response = generate(prompt, judge_model)
    
    if not llm_response.success:
        return {
            "scores": {},
            "overall_score": 0,
            "reasoning": "",
            "success": False,
            "error": llm_response.error
        }
    
    parsed = _parse_judge_json(llm_response.text)
    
    if not parsed:
        return {
            "scores": {},
            "overall_score": 0,
            "reasoning": "Failed to parse judge response",
            "raw_response": llm_response.text,
            "success": False,
            "error": "JSON parse error"
        }
    
    return {
        "scores": parsed.get("scores", {}),
        "overall_score": parsed.get("overall_score", 0),
        "reasoning": parsed.get("reasoning", ""),
        "success": True
    }


def run_position_bias_test(
    question: str,
    response_a: str,
    response_b: str,
    judge_model: str = JUDGE_MODEL
) -> dict:
    """
    Run both orderings to detect position bias.
    
    Args:
        question: The original question
        response_a: First response
        response_b: Second response
        judge_model: Model to use as judge
        
    Returns:
        Dict with both verdicts and bias analysis
    """
    # Run with original order (A, B)
    verdict_ab = judge_pairwise(
        question, response_a, response_b,
        randomize_order=False,
        judge_model=judge_model
    )
    
    # Run with swapped order (B, A)
    verdict_ba = judge_pairwise(
        question, response_b, response_a,
        randomize_order=False,
        judge_model=judge_model
    )
    # Correct the winner label since we swapped inputs
    if verdict_ba.winner == "A":
        corrected_winner_ba = "B"
    elif verdict_ba.winner == "B":
        corrected_winner_ba = "A"
    else:
        corrected_winner_ba = verdict_ba.winner
    
    # Check for flip
    consistent = verdict_ab.winner == corrected_winner_ba
    
    # Determine if there's position bias
    # If AB -> A wins and BA -> A wins (presented first), that's first-position bias
    first_won_ab = verdict_ab.winner == "A"  # A was first
    first_won_ba = verdict_ba.winner == "A"  # B was first, but judge said A (meaning first)
    
    return {
        "verdict_ab": {
            "winner": verdict_ab.winner,
            "order": "A first, B second",
            "success": verdict_ab.success
        },
        "verdict_ba": {
            "winner": corrected_winner_ba,
            "presented_winner": verdict_ba.winner,
            "order": "B first, A second",
            "success": verdict_ba.success
        },
        "consistent": consistent,
        "first_position_preference_ab": first_won_ab,
        "first_position_preference_ba": first_won_ba,
        "both_favor_first": first_won_ab and first_won_ba
    }
