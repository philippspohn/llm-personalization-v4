# Attribute response judging

For each `(user, generated_response)` pair produced by the
`attribute_responses` experiment, score the response on **every** response
attribute using the same judge as the `AttributePersonalizationSystem`
(`ParsedRatingJudge`, `openai/gpt-oss-120b`, thinking enabled).

Concretely, with `len(attributes) = 10` and both sides probed, each user has
20 responses (10 attrs × {follow, avoid}). Each response is scored against
all 20 `(attribute, side)` targets in the output — but the judge itself is
only called per `(response, attribute_name)` pair (10 calls per response),
because the avoid-side score is just `11 − follow_score`, exactly as
`PersonalizationAttributeJudge` derives it in the existing benchmark. So:

- **10 judge calls per response × 20 responses per user = 200 judge calls/user**
- **20 materialized score entries per response × 20 responses = 400 entries/user**

Output schema (one JSON per user, one user per line):

```json
{
  "user_id": "0",
  "split": "train",
  "gt_user_attributes": [...],
  "current_messages": [{"role": "user", "content": "..."}],
  "ratings": [
    {
      "gen_attribute": "casual",
      "gen_side": "follow",
      "response": "...",
      "scores": [
        {"attribute": "emotionally neutral", "side": "follow", "score": 4.3,
         "derived": false},
        {"attribute": "emotionally neutral", "side": "avoid",  "score": 6.7,
         "derived": true},
        ...   // 20 entries per response (10 attrs x {follow, avoid})
      ]
    },
    ...   // 20 entries (one per generated response)
  ]
}
```

`derived: true` flags the `avoid` rows that are computed as `11 − follow`,
not measured by an independent judge call.

The script is slurm-array friendly: it concatenates all input files in order
(tagging each user with the originating `split` name), then slices by
`(array_task_id, array_task_count)` and writes its slice to a per-task JSONL
file. Merging is `cat`.

## Inputs

You'll need:

- `data/attribute_responses/gpt-oss-120b_k10_1attrs_nemotron/train_4500.jsonl`
  (first 4500 lines of the merged `train.jsonl` produced by the
  `attribute_responses` experiment).
- `data/attribute_responses/gpt-oss-120b_k10_1attrs_nemotron/test.jsonl`
  (concatenation of the test parts).

Quick recipe:

```bash
cd data/attribute_responses/gpt-oss-120b_k10_1attrs_nemotron
cat train_part*_of0003.jsonl > train.jsonl
cat test_part*_of0001.jsonl  > test.jsonl
head -n 4500 train.jsonl     > train_4500.jsonl
```

## Submit

```bash
sbatch --array=0-7 sbatch/judge_attribute_responses.sbatch
```

After all 8 array tasks finish:

```bash
cat data/attribute_response_judging/<run>/ratings_part*_of*.jsonl \
    > data/attribute_response_judging/<run>/ratings.jsonl
```
