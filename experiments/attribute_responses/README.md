# Attribute responses

Generate one response per response-style attribute for every prompt in a synthetic
conversations dataset, using the same model + system-prompt template as the
`AttributePersonalizationSystem`.

For each user/prompt in the input JSONL we produce:

```
{
  "user_id": "0",
  "current_messages": [{"role": "user", "content": "..."}],
  "responses": [
    {"attribute": "emotionally neutral", "side": "follow", "response": "..."},
    {"attribute": "casual",              "side": "follow", "response": "..."},
    ...   // one per attribute (10 total for k=10)
  ]
}
```

The script is array-friendly: it slices the input by
`(SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT)` (or by the matching CLI
overrides) and writes its slice to a per-task JSONL file. Merging is just
concatenating the per-task files.

## Run locally

```bash
python experiments/attribute_responses/scripts/generate_attribute_responses.py \
    input_path=data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_train.jsonl \
    split_name=train \
    output_dir=data/attribute_responses/gpt-oss-120b_k10_1attrs_nemotron \
    array_task_id=0 array_task_count=1
```

## Response attribute set

The list of attributes is loaded from the `attributes` hydra config group in
`configs/attributes/{k2,k4,k6,k8,k10}.yaml`. Each file mirrors the matching
`data/attribute_selection/response_attributes/response_attributes_kN.yaml` and
the `response_attributes` block of the corresponding
`experiments/run_benchmark/configs/config_medium_attrs_kN_*attrs.yaml`.

Default is `attributes: k10`; override with e.g. `attributes=k4` on the CLI.

## Run as slurm array

The original k10 / 1-user-attr run uses
`sbatch/gen_attribute_responses_train.sbatch` and `..._test.sbatch`
(tensor_parallel_size=4, 4 GPUs/job).

For each of the alternative synthetic-conversation datasets in
`data/synthetic_conversations/gpt-oss-120b_kN_Mattrs_16000_1000_{train,test}.jsonl`
there is a dedicated sbatch script
`sbatch/gen_attribute_responses_kN_Mattrs_{train,test}.sbatch` that runs
Nemotron at tensor_parallel_size=2 across 2 GPUs and selects the matching
`attributes=kN` group. Train uses a 4-way array (`input_limit=9000`, ~2250
users / task); test uses a single task (full 1k pool).

```bash
sbatch --array=0-3 sbatch/gen_attribute_responses_k2_1attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k2_1attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k2_2attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k2_2attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k4_1attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k4_1attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k4_2attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k4_2attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k6_1attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k6_1attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k6_2attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k6_2attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k8_1attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k8_1attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k8_2attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k8_2attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k10_1attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k10_1attrs_test.sbatch
sbatch --array=0-3 sbatch/gen_attribute_responses_k10_2attrs_train.sbatch
sbatch --array=0-0 sbatch/gen_attribute_responses_k10_2attrs_test.sbatch
```

After all array tasks finish, merge with:

```bash
cat data/attribute_responses/<run>/train_part*_of*.jsonl \
    > data/attribute_responses/<run>/train.jsonl
cat data/attribute_responses/<run>/test_part*_of*.jsonl \
    > data/attribute_responses/<run>/test.jsonl
```
