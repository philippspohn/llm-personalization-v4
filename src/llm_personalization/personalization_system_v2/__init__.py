"""V2 personalization-system stack.

The v2 pipeline consumes pre-cached candidate responses (from
`experiments/attribute_responses`) and judge ratings (from
`experiments/attribute_response_judging`), so personalization methods at fit
time only need to read the JSONL artifacts on disk -- no LLM calls during
training other than what a method specifically needs (e.g. RAG-advanced "vibe"
descriptions, DPO LoRA fine-tuning).

See `experiments/run_benchmark_v2/README.md` for orchestration.
"""
