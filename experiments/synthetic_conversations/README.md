## Convert synthetic conversations JSONL to HF dataset

```bash
python experiments/synthetic_conversations/scripts/json_to_dataset.py \
    --train_jsonl data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_train.jsonl \
    --test_jsonl  data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_test.jsonl \
    --output_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_hf
```