"""
Configuration for evaluation experiments.
"""

# Model configuration for MacBook M4 with 15GB RAM
# Using models available in Ollama
CANDIDATE_MODEL = "llama3:8b-instruct-q4_K_M"
JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds

# Experiment configuration
DEFAULT_TEMPERATURE = 0.0  # For reproducibility
MAX_TOKENS = 1024

# File paths
TESTSET_PATH = "../testsets/experiments.jsonl"
RESULTS_DIR = "results"

# Experiment settings
EXPERIMENTS = {
    "structured_output": {
        "description": "Structured JSON output reliability",
        "task_types": ["extraction"],
        "metrics": ["json_valid", "required_fields", "latency"]
    },
    "prompt_iteration": {
        "description": "Before/after prompt engineering comparison",
        "task_types": ["instruction_following"],
        "metrics": ["format_compliance", "rubric_score"]
    },
    "position_bias": {
        "description": "LLM-as-judge position bias demonstration",
        "task_types": ["pairwise"],
        "metrics": ["position_preference", "flip_rate"]
    }
}
