# Model Router OpenEnv Simulation

A complete hackathon-ready OpenEnv environment simulating a real-world AI inference routing platform.

## Overview
As companies serve mixed AI workloads ranging from basic text summarization to multi-step reasoning over medical history, it isn't cost-effective to pass all tasks to the largest models (e.g. GPT-4). A model router evaluates each task and dynamically dispatches to `small`, `medium`, or `large` models optimizing the balance between **cost, latency, and quality** while never under-provisioning for high-risk contexts.

This project implements that routing logic as a standardized OpenEnv simulation environment where an Agent can be trained or evaluated on realistic tasks.

## Environment Design

- **States & Observations**: Powered by `Pydantic`. Features user request texts, latency bounds, and catalog indicators. The state dynamically reveals elements like exact task risk level after appropriate actions are taken.
- **Actions**:
  - `inspect_model_catalog`: Load catalog to discover model limits and costs.
  - `estimate_complexity_and_risk`: Provide reasoning to discern risk flags.
  - `choose_model`: Assign `small`, `medium`, or `large`.
  - `submit_route`: Close and submit the trace.
- **Grader & Reward Engine**: Highly normalized continuous score (`0.0` - `1.0`). Penalizes overspending on simple tasks, aggressively penalizes safety faults on high-risk tasks, and rewards logical sequencing (e.g. checking catalog before estimating).

## Tasks
1. **`easy_001`**: Basic grammar and spell check request. Low risk. Ideal model: `small`.
2. **`medium_001`**: Mid-sized coding data processing query. Modest complexity. Ideal model: `medium`.
3. **`hard_001`**: High-risk medical synthesis requiring de-identification logic. Must be given the `large` model.

## Running the Baseline

A robust script `inference.py` wraps the loop and routes decisions back via the OpenAI completions API. To run:

```sh
export OPENAI_API_KEY="your-key-here"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```
Outputs strict `[START]`, `[STEP]`, and `[END]` logging suitable for ingestion and metrics tracking.

## Deploying

This repository is built for Hugging Face Spaces & generic containers via Docker and exposes a FastAPI inference endpoint on port `7860`.

```sh
# Build
docker build -t model-router-env .

# Run
docker run -p 7860:7860 model-router-env
```
Endpoints: `/reset` and `/step` for standard remote HTTP environment loops.

## Validation target
- HF space answers HTTP
- `openenv validate` passes metadata
- Inference executes end-to-end
