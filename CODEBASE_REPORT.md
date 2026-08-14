# Codebase Report: CSE465 ColorBench Adaptive Skills (ACE Framework)

**Repository:** `cse465-project`  
**Active Branch:** `tanzim-ace`  
**Date:** August 2026  

---

## 1. Executive Summary & Purpose

This repository contains the implementation and experimentation pipeline for **CSE465**, focusing on evaluating and improving Vision-Language Model (VLM) reasoning on the **ColorBench** benchmark.

The core approach integrates the **ACE (Agentic Context Engineering)** framework, where a local unified VLM iteratively generates, tests, and refines domain-specific cognitive "skills" without modifying model weights.

### Foundational Literature
1. **ColorBench** (`umd-zhou-lab/ColorBench`):
   - Paper: *“COLORBENCH: Can VLMs See and Understand the Colorful World?”* ([2504.10514v3.md](file:///c:/Workspace/cse465-project/2504.10514v3.md))
   - Focus: Evaluates color perception, reasoning, optical illusions, mimicry, color blindness (Ishihara plates), counting, and negation.
2. **ACE Framework**:
   - Paper: *“Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models”* ([2510.04618v3.md](file:///c:/Workspace/cse465-project/2510.04618v3.md), ICLR 2026)
   - Focus: Modular context adaptation via generator, reflector, and solver roles to evolve playbooks and prevent context collapse.

---

## 2. System Architecture

The pipeline operates in two modes using a single local model ([`Qwen/Qwen2.5-VL-7B-Instruct`](file:///c:/Workspace/cse465-project/config.py#L9)) loaded with **4-bit NF4 Quantization** (BitsAndBytes) optimized for 15GB VRAM (e.g., Google Colab T4 GPU):

```
                       ┌─────────────────────────────────────┐
                       │   Input: Question, Options, Image   │
                       └──────────────────┬──────────────────┘
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        ▼                                                                   ▼
┌──────────────────┐                                       ┌───────────────────────────────────┐
│  Baseline Mode   │                                       │       Adaptive Skills (ACE)       │
│                  │                                       │                                   │
│ Direct VQA:      │                                       │ [Pass 1: Generator Agent]         │
│ Image + Question │                                       │ Classify type & craft Skill v1    │
│        │         │                                       │                │                  │
│        ▼         │                                       │                ▼                  │
│ Qwen2.5-VL Solver│                                       │ [Solve Pass 1]                    │
│        │         │                                       │ Mandated phrasing + Skill v1      │
│        ▼         │                                       │                │                  │
│    Prediction    │                                       │                ▼                  │
└──────────────────┘                                       │ [Pass 2+: Reflector Agent]        │
                                                           │ Analyze output trace -> Skill v2  │
                                                           │                │                  │
                                                           │                ▼                  │
                                                           │ [Solve Pass 2]                    │
                                                           │ Final Refined Prediction          │
                                                           └───────────────────────────────────┘
```

### Prompt Orchestration
- **Mandated Solver Phrasing:**
  > *"This is the skill to solve this question, now solve the question and give me the answer."*
- **Heuristic & JSON Fallback Stack:**
  `BaseAgent` employs a 4-tier JSON parsing cascade (Direct JSON parse → Markdown fence extraction → Outermost brace extraction → Key-value regex extraction → Rule-based semantic fallback).

---

## 3. Directory & File Manifest

| File / Path | Component | Description |
| :--- | :--- | :--- |
| [`config.py`](file:///c:/Workspace/cse465-project/config.py) | Configuration | Model ID, BitsAndBytes 4-bit config, dataset ID, default iterations and paths. |
| [`ace.py`](file:///c:/Workspace/cse465-project/ace.py) | Agent Orchestrator | Central `ACE` orchestrator coordinating `Generator` and `Reflector`. |
| [`core/base.py`](file:///c:/Workspace/cse465-project/core/base.py) | Core Agents | `BaseAgent` class with multi-tier JSON parsing, output normalization, and fallback heuristics. |
| [`core/generator.py`](file:///c:/Workspace/cse465-project/core/generator.py) | Core Agents | `Generator` agent producing initial cognitive skills for input questions. |
| [`core/reflector.py`](file:///c:/Workspace/cse465-project/core/reflector.py) | Core Agents | `Reflector` agent evaluating prior output traces to produce refined skills. |
| [`prompts/generator.py`](file:///c:/Workspace/cse465-project/prompts/generator.py) | Prompts | Directives and taxonomy categories (`color_negation`, `color_recognition`, `color_illusion`, etc.). |
| [`prompts/reflector.py`](file:///c:/Workspace/cse465-project/prompts/reflector.py) | Prompts | Directives for skill refinement and failure mode mitigation (verbosity, vagueness, etc.). |
| [`data_loader.py`](file:///c:/Workspace/cse465-project/data_loader.py) | Data & Logging | Streaming loader for HuggingFace `ColorBench` and crash-resilient `IncrementalLogger`. |
| [`qwen_solver.py`](file:///c:/Workspace/cse465-project/qwen_solver.py) | Inference Engine | 4-bit Qwen2.5-VL loader, option parser `(A)-(E)`, and explicit GPU memory cache clearers. |
| [`run_pipeline.py`](file:///c:/Workspace/cse465-project/run_pipeline.py) | Entrypoint | CLI execution script supporting `--task`, `--mode`, `--iterations`, and `--limit`. |
| [`eval_results.py`](file:///c:/Workspace/cse465-project/eval_results.py) | Evaluation | Accuracy calculator, pattern breakdown (Negation, Direct Object Color, Option E), and comparison script. |
| [`ColorBench_Adaptive_Skills.ipynb`](file:///c:/Workspace/cse465-project/ColorBench_Adaptive_Skills.ipynb) | Notebook | Interactive Colab/Jupyter notebook for running experiments. |
| [`results/`](file:///c:/Workspace/cse465-project/results) | Artifacts | Output `.jsonl` result logs per task and execution mode. |

---

## 4. Current Experimental Results

Evaluated on ColorBench subset (`Color Mimicry`):

| Result Log | Task | Mode | Accuracy |
| :--- | :--- | :--- | :--- |
| [`results_color_mimicry_baseline.jsonl`](file:///c:/Workspace/cse465-project/results/results_color_mimicry_baseline.jsonl) | Color Mimicry | Baseline (Direct VQA) | **10 / 15 (66.67%)** |
| [`results_color_mimicry_gen_verifier.jsonl`](file:///c:/Workspace/cse465-project/results/results_color_mimicry_gen_verifier.jsonl) | Color Mimicry | Gen + Verifier / Skills | **8 / 15 (53.33%)** |

---

## 5. Usage Commands

### Running Experiments
```bash
# Run baseline on Color Recognition (limit 20 samples)
python run_pipeline.py --task "Color Recognition" --mode baseline --limit 20

# Run 2-pass Adaptive Skills pipeline on Color Illusion
python run_pipeline.py --task "Color Illusion" --mode adaptive_skills --iterations 2
```

### Evaluating Results
```bash
# Print comparative report across all logs in results/
python eval_results.py

# Detailed report for a specific result file
python eval_results.py --file results/results_color_mimicry_baseline.jsonl
```
