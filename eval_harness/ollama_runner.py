"""
Ollama API wrapper with timing and error handling.
"""

import json
import time
import requests
from typing import Optional
from dataclasses import dataclass

from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT, DEFAULT_TEMPERATURE, MAX_TOKENS


@dataclass
class LLMResponse:
    """Container for LLM response with metadata."""
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: Optional[str] = None


def generate(
    prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    system_prompt: Optional[str] = None
) -> LLMResponse:
    """
    Generate a response from an Ollama model.
    
    Args:
        prompt: The user prompt
        model: Model name (e.g., "llama3.1:8b")
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt
        
    Returns:
        LLMResponse with text and metadata
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        result = response.json()
        
        return LLMResponse(
            text=result.get("response", ""),
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            success=True
        )
        
    except requests.exceptions.Timeout:
        return LLMResponse(
            text="",
            model=model,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error="Request timeout"
        )
    except requests.exceptions.RequestException as e:
        return LLMResponse(
            text="",
            model=model,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error=str(e)
        )


def chat(
    messages: list[dict],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS
) -> LLMResponse:
    """
    Generate a response using the chat API format.
    
    Args:
        messages: List of {"role": ..., "content": ...} dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        LLMResponse with text and metadata
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        result = response.json()
        
        return LLMResponse(
            text=result.get("message", {}).get("content", ""),
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            success=True
        )
        
    except requests.exceptions.RequestException as e:
        return LLMResponse(
            text="",
            model=model,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error=str(e)
        )


def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_models() -> list[str]:
    """List available models in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except requests.exceptions.RequestException:
        return []
