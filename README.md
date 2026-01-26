# When ``Better'' Prompts Hurt: Evaluation-Driven Iteration for LLM Applications

A comprehensive survey and reproducible experimental harness demonstrating that generic prompt improvements are not monotonic.

## Building the Paper

### Prerequisites

- LaTeX distribution (TeX Live, MacTeX, or MiKTeX)
- `latexmk` (included in most TeX distributions)

### Build Commands

**Using Make:**
```bash
make pdf
```

**Using latexmk directly:**
```bash
latexmk -pdf main.tex
```

**Clean build artifacts:**
```bash
make clean
```

## Project Structure

```
llm-eval-benchmarking-tex/
├── main.tex                    # Main document
├── sections/                   # All paper sections
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_quality_taxonomy.tex
│   ├── 03_eval_methods.tex
│   ├── 04_testset_design.tex
│   ├── 05_metrics_scoring.tex
│   ├── 06_rag_evaluation.tex
│   ├── 07_llm_judges.tex
│   ├── 08_case_studies.tex
│   ├── 09_failure_modes.tex
│   ├── 10_best_practices_checklist.tex
│   ├── 11_future_directions.tex
│   ├── 12_limitations.tex
│   └── 13_conclusion.tex
├── bib/
│   └── references.bib          # Bibliography
├── figures/
│   └── pipeline_overview.tex   # TikZ diagram
├── appendices/
│   ├── appendix_a_rubrics.tex
│   └── appendix_b_example_tests.tex
└── artifacts/                  # Standalone markdown templates
    ├── evaluation_checklist.md
    ├── rubric_template.md
    └── testset_template.md
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
ollama pull llama3.1:8b    # Candidate model
ollama pull qwen2.5:7b     # Judge model

# Install Python dependencies (using virtual environment)
cd eval_harness
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Experiments

```bash
cd eval_harness
source venv/bin/activate   # Activate virtual environment

# Run all experiments
python run_experiments.py --all

# Run individual experiments
python run_experiments.py --structured    # Exp 1: JSON output reliability
python run_experiments.py --prompt        # Exp 2: Prompt engineering impact  
python run_experiments.py --bias          # Exp 3: Position bias demonstration

# Dry run (without Ollama, for testing)
python run_experiments.py --all --dry-run --limit 5
```

### Experiment Details

| Experiment | Cases | Description |
|------------|-------|-------------|
| Structured Output | 20 | JSON validity rates with baseline vs. improved prompts |
| Prompt Iteration | 15 | Format compliance improvements from prompt engineering |
| Position Bias | 10 | Demonstrates ~10% first-position preference in LLM-as-judge |

Results are saved to `eval_harness/results/` as timestamped JSON files.

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
