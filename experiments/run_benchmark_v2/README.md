# run_benchmark_v2

A clean benchmark harness that consumes the **cached** candidate responses
(under `data/attribute_responses/`) and judge ratings (under
`data/attribute_response_judging/`), so personalization methods only need to
read JSONL artifacts at fit time -- no upstream LLM round-trips.

## Data flow

```
data/synthetic_conversations/<name>_{train,test}.jsonl    ┐
data/attribute_responses/<name>_<gen>/{train,test}.jsonl  │ -> prepare_dataset.py
data/attribute_response_judging/<...>/ratings_part*.jsonl ┘
        │
        ▼
data/benchmark_v2/<dataset_name>/{train,test,meta}.json{l}
        │
        ▼  run_benchmark.py
        │   1. method.fit(train)
        │   2. method.generate(test_inputs)             # one batched vLLM call
        │   3. UnpersonalizedBaseline.generate(...)      # one batched vLLM call
        │   4. judge.score(personalized + baseline)      # two batched judge calls
        ▼
outputs/benchmark_v2/<dataset_name>/<method_name>/<timestamp>/
    config.yaml                  # resolved hydra config
    results.jsonl                # per-user (response, score, baseline_response, baseline_score)
    summary.json                 # mean / std / win/tie/loss vs baseline
```

A v2 `PersonalizationExample` is the per-user record:
`(user_id, split, gt_user_attributes, history, prompt, candidates[2K])`,
where each `CandidateResponse` carries the original (attribute, side) it
was generated under, the response text, and `2K` ratings (one per
target (attribute, side); some flagged `derived` for the `11 - score`
opposite-side flips).

## One-time materialization

```bash
python experiments/run_benchmark_v2/scripts/prepare_dataset.py
# -> data/benchmark_v2/gpt-oss-120b_k10_1attrs_nemotron_gptoss_judge/{train,test,meta}.{jsonl,json}
# Add `force=true` to overwrite an existing materialized dataset.
# Pick a different dataset with `dataset=k4_1attrs` once that yaml exists.
```

For the current `k10_1attrs` cache this yields:

* 9000 train users (capped via `train_input_limit`, matches the upstream cap)
  - 9000 with the full 20 candidates
  - 4500 with full 20x20 ratings (judging only covered the first 4500)
* 1000 test users with the full 20 candidates **and** full 20x20 ratings

So routing-style methods that need ratings have ~4.5k usable train users.

## Run a method

```bash
python experiments/run_benchmark_v2/scripts/run_benchmark.py method=oracle_routing
python experiments/run_benchmark_v2/scripts/run_benchmark.py method=baseline
# Smoke test on 50 test users:
python experiments/run_benchmark_v2/scripts/run_benchmark.py method=oracle_routing test_limit=50
```

On Slurm:

```bash
# First runs (Qwen3.5-27B not yet cached -> needs huggingface.co):
sbatch sbatch/benchmark_v2.sbatch method=oracle_routing
sbatch sbatch/benchmark_v2.sbatch method=baseline
# Subsequent runs, once weights are cached, dodge node DNS flakiness:
HF_OFFLINE=1 sbatch sbatch/benchmark_v2.sbatch method=oracle_routing
```

## Hydra config groups

```
configs/
  config.yaml             # composes everything; defaults: dataset=k10_1attrs, method=oracle_routing
  prepare_dataset.yaml    # used only by prepare_dataset.py
  dataset/                # one yaml per cached (data, gen, judge) cell
    k10_1attrs.yaml
  attributes/             # response-attribute set (currently only consumed by prepare_dataset.py)
    k10.yaml
  method/                 # one yaml per personalization method
    baseline.yaml
    oracle_routing.yaml
  llm/qwen3_5_27b.yaml    # default generation model (used by all methods + baseline)
  llm/llama_3_3_70b.yaml  # alternative; gated repo (request access first)
  llm/nemotron.yaml       # alternative; the model used to produce the cached candidates
  judge/gpt-oss-120b.yaml # the eval judge
```

## Adding a new dataset cell

1. Wait until the corresponding `attribute_responses` and
   `attribute_response_judging` runs are complete.
2. Drop a new yaml under `configs/dataset/<cell>.yaml` (copy `k10_1attrs.yaml`
   and swap paths + `attributes_group`).
3. Drop a matching response-attribute list under `configs/attributes/<group>.yaml`
   if the group does not exist yet.
4. `python ... prepare_dataset.py dataset=<cell>` to materialize.
5. `sbatch sbatch/benchmark_v2.sbatch method=<m> dataset=<cell>` to evaluate.

## Methods (current and planned)

| method               | needs_gt_at_test | fit | notes                                          |
|----------------------|:----------------:|:---:|------------------------------------------------|
| `baseline`           | no               | -   | vanilla unpersonalized Nemotron call           |
| `oracle_routing`     | **yes**          | -   | injects GT (attr, side) into the system prompt |
| `routing` (planned)  | no               | yes | classifier on cached ratings                   |
| `rag_simple`         | no               | yes | embedding store of (history, prompt, best, worst) |
| `rag_advanced`       | no               | yes | LLM-derived "vibe" key + preference value      |
| `user_embedding_dpo` | no               | yes | DPO LoRA on cached preference pairs            |
