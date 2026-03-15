import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from scipy import stats

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.judge.parsed_rating_judge import ParsedRatingJudge
from llm_personalization.data.load_ultrachat import load_ultrachat_prompts


def load_nemotron_personas(n: int, seed: int = 42) -> list[str]:
    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    personas = []
    for row in ds:
        parts = [
            row.get("persona", ""),
            row.get("cultural_background", ""),
            row.get("skills_and_expertise", ""),
            row.get("hobbies_and_interests", ""),
        ]
        description = " ".join(p.strip() for p in parts if p.strip())
        if description:
            personas.append(description)
        if len(personas) == n:
            break
    return personas

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator. Your ONLY task is to rate how well an AI response is personalized \
and fitting for the provided user persona. IGNORE all other qualities of the response \
(factual accuracy, grammar, helpfulness in general, etc.).

Output a score on a scale from 1 to 10:
  1 = completely generic, could be addressed to anyone, ignores the persona entirely
  10 = perfectly tailored to this persona's background, interests, and communication style

Output only the numeric score, no other text."""

JUDGE_USER_TEMPLATE = """\
Please evaluate how well the AI response's writing style (word choice, sentence structure, tone, register, etc.) is personalized and fitting for the following user persona.

<persona>
{persona}
</persona>

<user_prompt>
{prompt}
</user_prompt>

<ai_response>
{response}
</ai_response>

How well is the response personalized and fitting for this specific persona?
Your evaluation (1-10):"""


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load personas ---
    print(f"Loading {cfg.num_personas} personas from Nemotron-Personas-USA...")
    personas = load_nemotron_personas(cfg.num_personas, seed=cfg.seed)
    print(f"Loaded {len(personas)} personas")

    # --- Load prompts ---
    print("Loading UltraChat prompts...")
    prompts = load_ultrachat_prompts("test_sft", limit=cfg.num_prompts, seed=cfg.seed)
    print(f"Loaded {len(prompts)} prompts")

    # --- Generate responses for each (model, system_prompt) combination ---
    system_prompts: dict[str, str] = OmegaConf.to_container(cfg.system_prompts, resolve=True)

    # all_responses keys: "{model_name}__{sp_name}" -> list[str]
    all_responses: dict[str, list[str]] = {}

    for model_name, model_cfg in cfg.llm_configs.items():
        print(f"[{model_name}] Loading model {model_cfg.model}...")
        helper: LLMHelper = hydra.utils.instantiate(model_cfg)
        helper.load()

        for sp_name, sp_text in system_prompts.items():
            run_key = f"{model_name}__{sp_name}"
            print(f"[{run_key}] Generating responses (system_prompt={sp_text!r})...")
            conversations = [
                ([{"role": "system", "content": sp_text}] if sp_text else [])
                + [{"role": "user", "content": p}]
                for p in prompts
            ]
            all_responses[run_key] = [r.content for r in helper.generate(conversations)]

        helper.unload()

    # --- Judge all (run_key x persona x prompt) combinations ---
    print("Loading judge model...")
    judge: ParsedRatingJudge = hydra.utils.instantiate(cfg.judge)
    judge.load()

    records = []
    judge_prompts = []

    for run_key, responses in all_responses.items():
        model_name, sp_name = run_key.split("__", 1)
        for persona_id, persona in enumerate(personas):
            for prompt_id, (prompt, response) in enumerate(zip(prompts, responses)):
                formatted = judge.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                            persona=persona,
                            prompt=prompt,
                            response=response,
                        )},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=judge.enable_thinking,
                )
                judge_prompts.append(formatted)
                records.append({
                    "model": model_name,
                    "system_prompt": sp_name,
                    "persona_id": persona_id,
                    "persona": persona,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": response,
                })

    n_runs = len(all_responses)
    total = len(judge_prompts)
    print(f"Judging {total} combinations ({n_runs} runs x {len(personas)} personas x {len(prompts)} prompts)...")
    scores = judge.judge_manual(judge_prompts)
    judge.unload()

    for record, score in zip(records, scores):
        record["score"] = score

    results_df = pd.DataFrame(records)
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # --- Summary ---
    def ci95(x):
        n = len(x)
        if n < 2:
            return float("nan")
        return stats.t.ppf(0.975, df=n - 1) * stats.sem(x)

    print("\n=== Mean personalization score by model x system_prompt ===")
    summary = results_df.groupby(["model", "system_prompt"])["score"].agg(
        mean="mean", ci95=ci95, count="count"
    )
    summary["mean ± 95% CI"] = summary.apply(
        lambda r: f"{r['mean']:.2f} ± {r['ci95']:.2f}", axis=1
    )
    print(summary[["mean ± 95% CI", "count"]].to_string())

    print("\n=== Mean personalization score by model x system_prompt x persona ===")
    persona_summary = (
        results_df.groupby(["model", "system_prompt", "persona_id"])["score"]
        .agg(mean="mean", ci95=ci95)
    )
    persona_summary["mean ± 95% CI"] = persona_summary.apply(
        lambda r: f"{r['mean']:.2f} ± {r['ci95']:.2f}", axis=1
    )
    pivot = persona_summary["mean ± 95% CI"].unstack(["model", "system_prompt"])
    pivot.index = [f"P{i:02d}" for i in pivot.index]
    print(pivot.to_string())


if __name__ == "__main__":
    main()
