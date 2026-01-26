#!/bin/bash
cd /Volumes/ExtSystem/Users/seraphic/Documents/dev/llm-eval-benchmarking
source .venv/bin/activate

echo "Starting Llama 3 Ablation..."
python eval_harness/run_ablation.py --dataset all --runs 5 --model llama3:8b-instruct-q4_K_M 2>&1 | tee results_ablation/llama3_log.txt

echo "Starting Qwen 2.5 Ablation..."
python eval_harness/run_ablation.py --dataset all --runs 5 --model qwen2.5:7b-instruct-q4_K_M 2>&1 | tee results_ablation/qwen25_log.txt

echo "All experiments complete at $(date)"
