# When ``Better'' Prompts Hurt: Evaluation-Driven Iteration for LLM Applications

A comprehensive survey and reproducible experimental harness demonstrating that generic prompt improvements are not monotonic.

## Building the Paper

### Prerequisites

- LaTeX distribution (TeX Live, MacTeX, or MiKTeX)
- `latexmk` (included in most TeX distributions)

## Build Commands

To build the paper:

```bash
cd paper
make pdf
```

## Project Structure

```
llm-eval-benchmarking/
├── paper/                      # LaTeX source
│   ├── main.tex
│   ├── sections/
│   ├── bib/
│   └── tables/
├── eval_harness/               # Reproducible experiments
│   ├── run_eval.py
│   ├── datasets/
│   └── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## How to Cite This Work

If you find this survey useful, please cite it as:

```bibtex
@article{llm-eval-survey-2026,
  title   = {When "Better" Prompts Hurt: Evaluation-Driven Iteration for LLM Applications},
  author  = {Commey, Daniel},
  year    = {2026},
  note    = {Survey/Tutorial Paper}
}
```

## Contents

1. **Introduction** – Why LLM evaluation differs from traditional software testing
2. **Quality Taxonomy** – Correctness, helpfulness, harmlessness, and groundedness
3. **Evaluation Methods** – Unit tests, golden sets, metamorphic testing, preference testing
4. **Test Set Design** – Creating representative datasets and adversarial prompts
5. **Metrics & Scoring** – Exact match, semantic similarity, calibration, inter-rater reliability
6. **RAG Evaluation** – Retrieval quality vs. generation quality, faithfulness metrics
7. **LLM-as-Judge** – When it works, when it fails, recommended guardrails
8. **Case Studies** – Customer support, RAG bots, summarization pipelines
9. **Failure Modes** – Prompt drift, format brittleness, silent regressions
10. **Best Practices Checklist** – Actionable guidance for practitioners
11. **Future Directions** – Tooling trends and standardization efforts
12. **Limitations** – What this guide does not cover
13. **Conclusion** – Summary and next steps
14. **Experimental Validation** – Original experiments demonstrating key evaluation methods

## Running the Evaluation Experiments

This paper includes an evaluation harness with reproducible experiments using local LLMs via Ollama.

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- MacBook M4 with 8-16GB RAM (or equivalent)

### Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull required models (in another terminal)
# Pull required models (in another terminal)
ollama pull llama3:8b      # Candidate model (v3.1 8B Instruct)
ollama pull qwen2.5:7b     # Judge model (v2.5 7B Instruct)

> **Note**: These exact model tags are the ones used in the paper experiments. Using different versions may yield different results.

# Install Python dependencies (using virtual environment)
cd eval_harness
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Experiments

```bash
cd eval_harness
source venv/bin/activate

# Run all experiments (Extraction, RAG, Instruction)
python run_eval.py --dataset all

# Run individual experiments
python run_eval.py --dataset extraction   # Experiment 1: JSON reliability
python run_eval.py --dataset rag          # Experiment 2: RAG grounding
python run_eval.py --dataset instruction  # Experiment 3: Prompt sensitivity
python run_eval.py --dataset bias         # Failure Analysis: Position bias

# Dry run (testing without Ollama)
python run_eval.py --dry-run --limit 2
```

### Experiment Details

| Experiment | Cases | Description |
|------------|-------|-------------|
| Structured Output | 20 | JSON validity rates with baseline vs. improved prompts |
| RAG Grounding | 15 | Citation accuracy and groundedness checks |
| Prompt Iteration | 15 | Format compliance improvements from prompt engineering |
| **Total** | **50** | **Main Experiment Suite** |
| Position Bias | 10 | (Separate) Demonstrates LLM-as-judge sensitivity |

Results are saved to `eval_harness/results/` as timestamped JSON files.
Ablation study logs are preserved in `eval_harness/results_ablation/`.

### Reproducing Paper Results

To regenerate the results tables in the paper:

```bash
cd eval_harness
python run_experiments.py --all
# Results saved to results/*.json
```

Expected runtime: ~30-60 minutes for all 50 test cases.

## License

This work is provided for educational and research purposes.

## AI Assistance Disclosure

This paper was developed with AI assistance for drafting and organizing content. The human author directed the outline, verified claims, and edited the final text.
