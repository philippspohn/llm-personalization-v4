"""Generate unpersonalized MC responses (no system prompt, no user history)
for the v3 robustness benchmark, once per (gen_model, robustness config).
Output is consumed by `personalization_system_v3.benchmark` so per-method
robustness runs can skip the unpers pass entirely.

Output path:
    data/unpersonalized_robustness/{gen_model_tag}/{cache_key}.jsonl
where cache_key encodes the include flags + per-source limits + seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from llm_personalization.benchmark.robustness_benchmark.robustness_dataset import (
    format_mc_prompt,
    load_robustness_questions,
    parse_answer_letter,
)
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.personalization_system_v3.robustness import unpers_cache_key


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())

    print("[unpers_robustness] loading questions...")
    questions = load_robustness_questions(
        include_mmlu_pro=cfg.robustness.get("include_mmlu_pro", True),
        include_truthfulqa=cfg.robustness.get("include_truthfulqa", True),
        include_bbq=cfg.robustness.get("include_bbq", False),
        mmlu_pro_limit=cfg.robustness.get("mmlu_pro_limit", None),
        truthfulqa_limit=cfg.robustness.get("truthfulqa_limit", None),
        bbq_limit=cfg.robustness.get("bbq_limit", None),
        seed=cfg.robustness.get("seed", 42),
    )
    print(f"[unpers_robustness] loaded {len(questions)} questions")

    cache_key = unpers_cache_key(cfg.robustness)
    out_dir = project_root / "data" / "unpersonalized_robustness" / cfg.gen_model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cache_key}.jsonl"

    if out_path.exists() and not cfg.get("force", False):
        print(f"[unpers_robustness] {out_path} already exists; skipping (pass force=true to regenerate)")
        return

    prompts = [[{"role": "user", "content": format_mc_prompt(q)}] for q in questions]
    print(f"[unpers_robustness] generating {len(prompts)} responses with {cfg.llm.model}")
    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()
    try:
        responses = [r.content for r in llm.generate(prompts)]
    finally:
        llm.unload()

    # Per-source (and BBQ per-category) accuracy breakdown.
    src_correct: dict = {}
    src_total: dict = {}
    bbq_cat_correct: dict = {}
    bbq_cat_total: dict = {}
    with open(out_path, "w") as f:
        for q, r in zip(questions, responses):
            parsed = parse_answer_letter(r)
            is_correct = parsed == q.correct_letter
            f.write(json.dumps({
                "question_id": q.question_id,
                "source": q.source,
                "category": (q.metadata or {}).get("category"),
                "response": r,
                "parsed_letter": parsed,
                "correct_letter": q.correct_letter,
            }) + "\n")
            src_total[q.source] = src_total.get(q.source, 0) + 1
            if is_correct:
                src_correct[q.source] = src_correct.get(q.source, 0) + 1
            if q.source == "bbq":
                cat = (q.metadata or {}).get("category", "unknown")
                bbq_cat_total[cat] = bbq_cat_total.get(cat, 0) + 1
                if is_correct:
                    bbq_cat_correct[cat] = bbq_cat_correct.get(cat, 0) + 1

    print(f"[unpers_robustness] wrote {len(responses)} records to {out_path}")
    print(f"[unpers_robustness] per-source accuracy:")
    for src in sorted(src_total):
        c, t = src_correct.get(src, 0), src_total[src]
        print(f"    {src}: {c}/{t} = {c/t:.4f}")
    if bbq_cat_total:
        print(f"[unpers_robustness] bbq per-category accuracy:")
        for cat in sorted(bbq_cat_total):
            c, t = bbq_cat_correct.get(cat, 0), bbq_cat_total[cat]
            print(f"    bbq/{cat}: {c}/{t} = {c/t:.4f}")


if __name__ == "__main__":
    main()
