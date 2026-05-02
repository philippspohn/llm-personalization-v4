"""Judge all generated responses under multiple judge criteria.

Criteria:
  - overall_quality: "how good is this response?"
  - personalization_ungrounded: "how personalized is this response?" (no grounding)
  - personalization_attr_grounded: personalization given a random attribute profile per prompt
  - personalization_persona_grounded: personalization given a nemotron persona per prompt
  - attribute_compliance_<attr>: for each of the k10 attributes, spectrum score
        (no "better" connotation — pure attribute compliance)

All judge prompts are built, then judged in one big vLLM batch, then scores
are saved to judge_scores.jsonl. Finally, a summary table is printed/saved.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

from llm_personalization.data.load_nemotron_personas import load_nemotron_personas
from llm_personalization.judge.parsed_rating_judge import ParsedRatingJudge


# ---------- Judge prompt templates ---------- #

SYS_SPECTRUM = (
    "You are an expert, impartial evaluator. Your ONLY task is to rate the response "
    "on the given spectrum. Neither end of the spectrum is inherently better — "
    "output where on the spectrum this response sits.\n\n"
    "Output a score on a scale from 1 (clearly exhibits the OPPOSITE of the attribute) "
    "to 100 (clearly exhibits the attribute).\n\n"
    "First, use internal thinking to reason through the evaluation step-by-step. "
    "Your final visible output must be ONLY the numeric score (1-100)."
)

SYS_QUALITY = (
    "You are an expert, impartial evaluator. Rate the overall quality of the AI "
    "response to the user's prompt. Consider clarity, correctness, depth, and usefulness.\n\n"
    "Output a score on a scale from 1 (very low quality) to 100 (excellent quality).\n\n"
    "First, use internal thinking to reason through the evaluation. "
    "Your final visible output must be ONLY the numeric score (1-100)."
)

SYS_PERSONALIZATION_UNGROUNDED = (
    "You are an expert, impartial evaluator. Your ONLY task is to rate how personalized "
    "the AI response is — that is, how clearly it appears tailored to a specific user "
    "rather than written generically for anyone. IGNORE all other qualities "
    "(factual accuracy, grammar, helpfulness in general, etc.).\n\n"
    "Output a score on a scale from 1 (completely generic, could be addressed to anyone) "
    "to 100 (highly personalized, clearly tailored to a specific user).\n\n"
    "First, use internal thinking to reason through the evaluation. "
    "Your final visible output must be ONLY the numeric score (1-100)."
)

SYS_PERSONALIZATION_GROUNDED = (
    "You are an expert, impartial evaluator. Your ONLY task is to rate how well the AI "
    "response is personalized for the provided user profile. IGNORE all other qualities.\n\n"
    "Output a score on a scale from 1 (completely generic / ignores the profile) "
    "to 100 (perfectly tailored to this specific profile).\n\n"
    "First, use internal thinking to reason through the evaluation. "
    "Your final visible output must be ONLY the numeric score (1-100)."
)

USR_QUALITY = """Please rate the overall quality of the AI response to the user's prompt.

<user_prompt>
{prompt}
</user_prompt>
<ai_response>
{response}
</ai_response>

Overall quality (1-100):"""

USR_PERSONALIZATION_UNGROUNDED = """Please rate how personalized the AI response is (tailored to a specific user vs. generic).

<user_prompt>
{prompt}
</user_prompt>
<ai_response>
{response}
</ai_response>

Level of personalization (1-100):"""

USR_PERSONALIZATION_ATTR = """Please rate how well the AI response is personalized for a user with the following preferred response attributes.

<user_preferred_attributes>
{attributes}
</user_preferred_attributes>
<user_prompt>
{prompt}
</user_prompt>
<ai_response>
{response}
</ai_response>

How well-personalized is the response for this user (1-100):"""

USR_PERSONALIZATION_PERSONA = """Please rate how well the AI response is personalized for the following user persona.

<user_persona>
{persona}
</user_persona>
<user_prompt>
{prompt}
</user_prompt>
<ai_response>
{response}
</ai_response>

How well-personalized is the response for this persona (1-100):"""

USR_ATTR_COMPLIANCE = """Please rate where the AI response falls on the spectrum for the following attribute. Neither end is better — we simply want to know where on the spectrum it sits.

<attribute>
{attribute}
</attribute>
<user_prompt>
{prompt}
</user_prompt>
<ai_response>
{response}
</ai_response>

Score (1 = opposite of attribute, 100 = strongly exhibits attribute):"""


# ---------- Helpers ---------- #

def build_judge_prompt(judge: ParsedRatingJudge, system: str, user: str) -> str:
    return judge.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=judge.enable_thinking,
    )


def sample_grounding_attrs(attrs: list[str], k: int, seed: int, prompt_id: int) -> list[str]:
    rng = random.Random(f"{seed}-{prompt_id}")
    return rng.sample(attrs, k)


@hydra.main(version_base=None, config_path="../configs", config_name="config_judge")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / cfg.responses_filename
    scores_path = output_dir / cfg.scores_filename
    summary_path = output_dir / cfg.summary_filename

    # --- Load responses --- #
    print(f"Loading responses from {responses_path}")
    records = [json.loads(line) for line in responses_path.open()]
    print(f"Loaded {len(records)} responses")

    # --- Load attribute list --- #
    with open(cfg.k10_attributes_path) as f:
        attributes: list[str] = yaml.safe_load(f)["candidate_attributes"]

    # --- Load personas (one per unique prompt_id) --- #
    unique_prompt_ids = sorted({r["prompt_id"] for r in records})
    print(f"Loading {len(unique_prompt_ids)} nemotron personas (split={cfg.persona_split})...")
    personas_raw = load_nemotron_personas(
        split=cfg.persona_split, limit=max(unique_prompt_ids) + 1, seed=cfg.seed
    )
    personas_by_pid: dict[int, str] = {}
    for pid in unique_prompt_ids:
        p = personas_raw[pid]
        parts = [p.get("persona", ""), p.get("skills_and_expertise", ""), p.get("hobbies_and_interests", "")]
        personas_by_pid[pid] = " ".join(x.strip() for x in parts if x and x.strip())

    # --- Precompute per-prompt grounding --- #
    attrs_by_pid: dict[int, list[str]] = {
        pid: sample_grounding_attrs(attributes, cfg.num_grounding_attributes, cfg.seed, pid)
        for pid in unique_prompt_ids
    }

    # --- Load judge --- #
    print("Loading judge model...")
    judge: ParsedRatingJudge = hydra.utils.instantiate(cfg.judge)
    judge.load()

    # --- Build all judge prompts --- #
    judge_prompts: list[str] = []
    score_meta: list[dict] = []  # one entry per judge prompt

    for i, r in enumerate(records):
        pid = r["prompt_id"]
        prompt = r["prompt"]
        response = r["response"]

        base_meta = {
            "record_idx": i,
            "model": r["model"],
            "system_prompt_variant": r["system_prompt_variant"],
            "prompt_id": pid,
        }

        # overall_quality
        judge_prompts.append(build_judge_prompt(
            judge, SYS_QUALITY, USR_QUALITY.format(prompt=prompt, response=response)
        ))
        score_meta.append({**base_meta, "criterion": "overall_quality", "grounding": None})

        # personalization_ungrounded
        judge_prompts.append(build_judge_prompt(
            judge, SYS_PERSONALIZATION_UNGROUNDED,
            USR_PERSONALIZATION_UNGROUNDED.format(prompt=prompt, response=response),
        ))
        score_meta.append({**base_meta, "criterion": "personalization_ungrounded", "grounding": None})

        # personalization_attr_grounded
        attr_profile = attrs_by_pid[pid]
        judge_prompts.append(build_judge_prompt(
            judge, SYS_PERSONALIZATION_GROUNDED,
            USR_PERSONALIZATION_ATTR.format(
                attributes=", ".join(attr_profile), prompt=prompt, response=response,
            ),
        ))
        score_meta.append({
            **base_meta,
            "criterion": "personalization_attr_grounded",
            "grounding": ", ".join(attr_profile),
        })

        # personalization_persona_grounded
        persona = personas_by_pid[pid]
        judge_prompts.append(build_judge_prompt(
            judge, SYS_PERSONALIZATION_GROUNDED,
            USR_PERSONALIZATION_PERSONA.format(persona=persona, prompt=prompt, response=response),
        ))
        score_meta.append({
            **base_meta,
            "criterion": "personalization_persona_grounded",
            "grounding": persona,
        })

        # attribute_compliance_<attr> — one per attribute
        for attr in attributes:
            judge_prompts.append(build_judge_prompt(
                judge, SYS_SPECTRUM,
                USR_ATTR_COMPLIANCE.format(attribute=attr, prompt=prompt, response=response),
            ))
            score_meta.append({
                **base_meta,
                "criterion": f"attribute_compliance__{attr.replace(' ', '_')}",
                "grounding": attr,
            })

    print(f"Built {len(judge_prompts)} judge prompts. Running judge...")
    scores = judge.judge_manual(judge_prompts)
    judge.unload()

    # --- Save scores --- #
    with scores_path.open("w") as f:
        for meta, score in zip(score_meta, scores):
            f.write(json.dumps({**meta, "score": score}) + "\n")
    print(f"Saved scores to {scores_path}")

    # --- Summary: mean score per (model, system_prompt_variant, criterion) --- #
    # attribute_compliance_* criteria are aggregated separately per attribute.
    df = pd.DataFrame([{**m, "score": s} for m, s in zip(score_meta, scores)])

    non_attr_df = df[~df["criterion"].str.startswith("attribute_compliance__")]
    summary_non_attr = (
        non_attr_df
        .groupby(["model", "system_prompt_variant", "criterion"])["score"]
        .mean()
        .unstack("criterion")
    )

    attr_df = df[df["criterion"].str.startswith("attribute_compliance__")]
    summary_attr = (
        attr_df
        .groupby(["model", "system_prompt_variant", "criterion"])["score"]
        .mean()
        .unstack("criterion")
    )

    summary = pd.concat([summary_non_attr, summary_attr], axis=1)
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")
    print("\n=== Summary (mean score per variant x criterion) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
