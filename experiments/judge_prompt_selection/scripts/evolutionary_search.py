import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
from vllm import SamplingParams

from llm_personalization.judge.parsed_rating_judge import ParsedRatingJudge
from llm_personalization.judge.prompt_templates import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_THINKING,
    JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_mutation_output(raw: str) -> str:
    """
    Extract the usable system prompt from a raw mutator LLM output.

    gpt-oss emits `analysis<thinking>assistantfinal<content>` regardless of the
    enable_thinking flag. We keep only the content after `assistantfinal`, and
    strip surrounding <system_prompt>...</system_prompt> tags if present.
    """
    text = raw
    af_pos = text.find("assistantfinal")
    if af_pos != -1:
        text = text[af_pos + len("assistantfinal"):]
    # Also guard against <think>...</think> formats.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"<system_prompt>(.*?)</system_prompt>", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    return text.strip()


def ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character-level n-gram Jaccard similarity."""
    def ngrams(text):
        return set(text[i:i + n] for i in range(len(text) - n + 1))

    a_set = ngrams(a.lower())
    b_set = ngrams(b.lower())
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


def build_eval_set(
    df: pd.DataFrame,
    attributes: list[str],
    num_prompts: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Sample a fixed evaluation set that is reused across all iterations.

    Returns:
        follow_df  – rows from df where side == "follow"
        pids       – array of sampled prompt_ids
        gt_attrs   – mapping pid -> ground-truth attribute
    """
    rng = np.random.default_rng(seed)
    follow_df = df[df["side"] == "follow"].copy()
    prompt_ids = np.array(sorted(follow_df["prompt_id"].unique()))
    pids = rng.choice(prompt_ids, size=min(num_prompts, len(prompt_ids)), replace=False)
    gt_attrs = {int(pid): str(rng.choice(attributes)) for pid in pids}
    return follow_df, pids, gt_attrs


def evaluate_system_prompt(
    judge: ParsedRatingJudge,
    follow_df: pd.DataFrame,
    attributes: list[str],
    sampled_pids: np.ndarray,
    gt_attrs: dict,
    system_prompt: str,
) -> tuple[float, float]:
    """
    Score a candidate system prompt.

    For each (prompt_id, gt_attribute) pair in the eval set:
      - Retrieve the response generated with gt_attribute
      - Judge it against every attribute using the candidate system_prompt
      - margin = score(gt_attr) - mean(score(all_attrs))
      - top1  = 1 if argmax over eval_attrs equals gt_attr else 0

    Returns (mean margin, top-1 accuracy) across all pairs.
    """
    judge_prompts: list[str] = []
    metadata: list[dict] = []

    for pid in sampled_pids:
        pid = int(pid)
        gt_attr = gt_attrs[pid]
        rows = follow_df[
            (follow_df["prompt_id"] == pid) & (follow_df["attribute"] == gt_attr)
        ]
        if rows.empty:
            print(f"[evaluate] WARNING: no row for pid={pid}, attr={gt_attr}")
            continue
        row = rows.iloc[0]

        conv_str = f"<message role='user'>{row['prompt']}</message>\n"
        for eval_attr in attributes:
            user_prompt = JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE.format(
                conversation=conv_str,
                response=row["response"],
                attribute=eval_attr,
            )
            full_prompt = judge.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=judge.enable_thinking,
            )
            judge_prompts.append(full_prompt)
            metadata.append({"pid": pid, "gt_attr": gt_attr, "eval_attr": eval_attr})

    if not judge_prompts:
        return 0.0, 0.0

    raw_scores = judge.judge_manual(judge_prompts)

    results = pd.DataFrame(metadata)
    results["score"] = [s if s is not None else 5.5 for s in raw_scores]

    margins: list[float] = []
    top1_hits: list[int] = []
    for (pid, gt_attr), group in results.groupby(["pid", "gt_attr"]):
        gt_rows = group[group["eval_attr"] == gt_attr]["score"]
        if gt_rows.empty:
            continue
        margins.append(float(gt_rows.values[0] - group["score"].mean()))
        best_row = group.loc[group["score"].idxmax()]
        top1_hits.append(int(best_row["eval_attr"] == gt_attr))

    if not margins:
        return 0.0, 0.0
    return float(np.mean(margins)), float(np.mean(top1_hits))


# ---------------------------------------------------------------------------
# Mutation  (reuses the already-loaded judge LLM — no extra model loading)
# ---------------------------------------------------------------------------

MUTATE_SYSTEM_PROMPT = (
    "You are an optimization assistant helping improve system prompts for an LLM judge.\n"
    "The judge's task: rate (on a scale 1-100) how strongly an AI response exhibits a "
    "specific attribute.\n"
    "The goal is to maximize discriminability: a response generated *for* the attribute "
    "should score much higher than responses generated for *other* attributes."
)

MUTATE_USER_TEMPLATE = """\
Here is the current judge system prompt:

<system_prompt>
{system_prompt}
</system_prompt>

Its discriminability score is {score:.4f} (scale ~0-5; higher means the judge is better at \
distinguishing matching responses from non-matching ones).

Create a meaningfully different variation of this system prompt that might achieve better \
discriminability. Keep the same purpose (rating attribute presence on a 1-100 scale) but try a \
different framing, emphasis, or instruction strategy.

Output ONLY the new system prompt text — no preamble, explanation, or formatting.\
"""


def generate_mutations(
    judge: ParsedRatingJudge,
    mutator_sampling_params: dict,
    candidates: list[tuple[str, float, float]],
    num_mutations: int,
    rng: np.random.Generator,
    selection_temperature: float,
) -> list[str]:
    """
    Select parents from the pool (softmax-weighted by score) and generate
    one mutation per parent using the already-loaded judge LLM.
    Thinking is disabled for mutation — it's just a regular generation task.
    """
    scores_arr = np.array([s for _, s, _ in candidates], dtype=float)
    shifted = scores_arr - scores_arr.max()
    weights = np.exp(shifted / max(selection_temperature, 1e-6))
    weights /= weights.sum()

    parent_indices = rng.choice(len(candidates), size=num_mutations, p=weights, replace=True)

    raw_prompts = []
    for idx in parent_indices:
        parent_prompt, parent_score, _ = candidates[idx]
        raw_prompt = judge.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MUTATE_SYSTEM_PROMPT},
                {"role": "user", "content": MUTATE_USER_TEMPLATE.format(
                    system_prompt=parent_prompt,
                    score=parent_score,
                )},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_prompts.append(raw_prompt)

    outputs = judge.llm.generate(raw_prompts, sampling_params=SamplingParams(**mutator_sampling_params))
    return [parse_mutation_output(o.outputs[0].text) for o in outputs]


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

def try_add_to_pool(
    candidates: list[tuple[str, float, float]],
    new_prompt: str,
    new_score: float,
    new_accuracy: float,
    max_pool_size: int,
    similarity_threshold: float,
    ngram_n: int = 3,
) -> tuple[bool, str]:
    """
    Attempt to insert (new_prompt, new_score, new_accuracy) into candidates.
    Returns (was_added, reason_string). Selection is driven by margin score
    (new_score); accuracy is carried alongside for reporting.

    Rules (in order):
      1. If there is a candidate with Jaccard ≥ similarity_threshold:
         replace it only if new_score is strictly better.
      2. Else if pool is not full: append.
      3. Else replace the lowest-scoring candidate if new_score is better.
    """
    most_similar_idx: int | None = None
    max_sim = 0.0
    for i, (cand_prompt, _, _) in enumerate(candidates):
        sim = ngram_jaccard(new_prompt, cand_prompt, ngram_n)
        if sim > max_sim:
            max_sim = sim
            most_similar_idx = i

    if most_similar_idx is not None and max_sim >= similarity_threshold:
        _, existing_score, _ = candidates[most_similar_idx]
        if new_score > existing_score:
            candidates[most_similar_idx] = (new_prompt, new_score, new_accuracy)
            return True, f"replaced similar (sim={max_sim:.3f}, {existing_score:.4f} → {new_score:.4f})"
        return False, f"discarded (similar to #{most_similar_idx}, sim={max_sim:.3f}, score {new_score:.4f} ≤ {existing_score:.4f})"

    if len(candidates) < max_pool_size:
        candidates.append((new_prompt, new_score, new_accuracy))
        return True, f"appended (pool size now {len(candidates)})"

    min_idx = int(np.argmin([s for _, s, _ in candidates]))
    min_score = candidates[min_idx][1]
    if new_score > min_score:
        candidates[min_idx] = (new_prompt, new_score, new_accuracy)
        return True, f"replaced worst (#{min_idx}, {min_score:.4f} → {new_score:.4f})"
    return False, f"discarded (score {new_score:.4f} ≤ pool minimum {min_score:.4f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="evolutionary_search")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir.replace("{timestamp}", timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.get("seed", 42))

    # ------------------------------------------------------------------
    # Load data + build fixed eval set
    # ------------------------------------------------------------------
    df = pd.read_csv(cfg.input_path)
    attributes = list(cfg.attributes)
    print(f"[EvoSearch] Loaded {len(df)} rows, {len(attributes)} attributes")

    follow_df, sampled_pids, gt_attrs = build_eval_set(
        df, attributes, cfg.num_eval_prompts, cfg.get("eval_seed", 0)
    )
    print(f"[EvoSearch] Eval set: {len(sampled_pids)} prompts, seed={cfg.get('eval_seed', 0)}")

    # ------------------------------------------------------------------
    # Load the judge once — it stays loaded for the entire run.
    # Mutations reuse judge.llm directly, avoiding repeated load/unload.
    # ------------------------------------------------------------------
    judge: ParsedRatingJudge = hydra.utils.instantiate(cfg.judge)
    print("[EvoSearch] Loading judge (stays loaded for full run)...")
    judge.load()

    # Seed the pool with the existing prompt from prompt_templates.py.
    # Use the thinking variant when the judge has enable_thinking=True.
    seed_prompt = JUDGE_SYSTEM_PROMPT_THINKING if judge.enable_thinking else JUDGE_SYSTEM_PROMPT
    print(f"[EvoSearch] Evaluating seed prompt (thinking={judge.enable_thinking})...")
    initial_score, initial_acc = evaluate_system_prompt(
        judge, follow_df, attributes, sampled_pids, gt_attrs, seed_prompt
    )
    print(f"[EvoSearch] Seed margin: {initial_score:.4f}  top1: {initial_acc:.3f}")

    candidates: list[tuple[str, float, float]] = [(seed_prompt, initial_score, initial_acc)]

    history: list[dict] = [{
        "batch": 0,
        "prompt": seed_prompt,
        "score": initial_score,
        "accuracy": initial_acc,
        "added": True,
        "reason": "initial seed",
    }]

    # ------------------------------------------------------------------
    # Evolutionary loop
    # ------------------------------------------------------------------
    num_batches: int = cfg.num_iterations
    batch_size: int = cfg.get("batch_size", 5)
    max_pool_size: int = cfg.get("max_pool_size", 10)
    similarity_threshold: float = cfg.get("similarity_threshold", 0.5)
    selection_temperature: float = cfg.get("selection_temperature", 1.0)
    mutator_sampling_params: dict = dict(cfg.get("mutator_sampling_params", {}))

    for batch_idx in range(1, num_batches + 1):
        print(f"\n[EvoSearch] ── Batch {batch_idx}/{num_batches} ──")

        # ── Mutation phase (uses judge.llm directly, no extra loading) ──
        print(f"[EvoSearch]   Generating {batch_size} mutations...")
        new_prompts = generate_mutations(
            judge, mutator_sampling_params, candidates, batch_size, rng, selection_temperature
        )

        # ── Evaluation phase ─────────────────────────────────────────────
        new_results: list[tuple[float, float]] = []
        for i, new_prompt in enumerate(new_prompts):
            print(f"[EvoSearch]   Evaluating mutation {i + 1}/{batch_size}...")
            score, accuracy = evaluate_system_prompt(
                judge, follow_df, attributes, sampled_pids, gt_attrs, new_prompt
            )
            new_results.append((score, accuracy))
            print(f"[EvoSearch]   margin: {score:.4f}  top1: {accuracy:.3f}")

        # ── Pool update ──────────────────────────────────────────────────
        for new_prompt, (new_score, new_acc) in zip(new_prompts, new_results):
            added, reason = try_add_to_pool(
                candidates, new_prompt, new_score, new_acc, max_pool_size, similarity_threshold
            )
            print(f"[EvoSearch]   {reason}")
            history.append({
                "batch": batch_idx,
                "prompt": new_prompt,
                "score": new_score,
                "accuracy": new_acc,
                "added": added,
                "reason": reason,
            })

        # ── Checkpoint ──────────────────────────────────────────────────
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        sorted_cands = sorted(candidates, key=lambda x: -x[1])
        with open(output_dir / "candidates.json", "w") as f:
            json.dump(
                [{"prompt": p, "score": s, "accuracy": a} for p, s, a in sorted_cands],
                f, indent=2,
            )

        best_prompt, best_score, best_acc = sorted_cands[0]
        print(f"[EvoSearch]   Pool: {len(candidates)} candidates | best margin: {best_score:.4f}  top1: {best_acc:.3f}")

    judge.unload()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    best_prompt, best_score, best_acc = max(candidates, key=lambda x: x[1])
    print(f"\n[EvoSearch] Done! Best margin: {best_score:.4f}  top1: {best_acc:.3f}")
    print(f"[EvoSearch] Best system prompt:\n{best_prompt}")
    print(f"[EvoSearch] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
