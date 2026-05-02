"""Generate responses for experiment 1 of persona_judge_bias_v2.

For each (model, system_prompt_variant, prompt) triple, generate one response
and append it as a JSONL record. Each model is loaded once and generates all
(variant x prompt) pairs in a single vLLM batch, then unloaded.
"""
import json
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from llm_personalization.data.load_ultrachat import load_ultrachat_prompts
from llm_personalization.llm.llm_helper import LLMHelper


def build_system_prompts(
    system_prompts_cfg: dict,
    attributes: list[str],
) -> dict[str, str]:
    """Assemble the full {variant_name -> system_prompt_text} mapping."""
    prompts: dict[str, str] = {}
    for name, text in system_prompts_cfg["static"].items():
        prompts[name] = text
    for attr in attributes:
        key = attr.replace(" ", "_")
        prompts[f"follow_{key}"] = system_prompts_cfg["follow_template"].format(attribute=attr)
        prompts[f"avoid_{key}"] = system_prompts_cfg["avoid_template"].format(attribute=attr)
    return prompts


@hydra.main(version_base=None, config_path="../configs", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / cfg.responses_filename

    with open(cfg.k10_attributes_path) as f:
        attributes = yaml.safe_load(f)["candidate_attributes"]
    with open(cfg.system_prompts_path) as f:
        sp_cfg = yaml.safe_load(f)

    system_prompts = build_system_prompts(sp_cfg, attributes)
    variant_names = list(system_prompts.keys())
    print(f"System prompt variants ({len(variant_names)}): {variant_names}")

    print(f"Loading {cfg.num_prompts} UltraChat prompts...")
    prompts = load_ultrachat_prompts("test_sft", limit=cfg.num_prompts, seed=cfg.seed)
    print(f"Loaded {len(prompts)} prompts")

    # Truncate / fresh start
    responses_path.write_text("")
    print(f"Writing responses to {responses_path}")

    for model_name, model_cfg in cfg.llm_configs.items():
        print(f"\n=== [{model_name}] Loading {model_cfg.model} ===")
        helper: LLMHelper = hydra.utils.instantiate(model_cfg)
        helper.load()

        conversations: list[list[dict[str, str]]] = []
        meta: list[tuple[str, int]] = []  # (variant_name, prompt_id)
        for variant_name, sp_text in system_prompts.items():
            for prompt_id, prompt in enumerate(prompts):
                conv = []
                if sp_text:
                    conv.append({"role": "system", "content": sp_text})
                conv.append({"role": "user", "content": prompt})
                conversations.append(conv)
                meta.append((variant_name, prompt_id))

        print(f"[{model_name}] Generating {len(conversations)} responses "
              f"({len(variant_names)} variants x {len(prompts)} prompts)...")
        responses = helper.generate(conversations)

        with responses_path.open("a") as f:
            for (variant_name, prompt_id), resp in zip(meta, responses):
                record = {
                    "model": model_name,
                    "model_hf": model_cfg.model,
                    "system_prompt_variant": variant_name,
                    "prompt_id": prompt_id,
                    "prompt": prompts[prompt_id],
                    "response": resp.content,
                    "finish_reason_stop": resp.finish_reason_stop,
                }
                f.write(json.dumps(record) + "\n")
        print(f"[{model_name}] Appended {len(responses)} records to {responses_path}")

        helper.unload()

    print(f"\nDone. All responses saved to {responses_path}")


if __name__ == "__main__":
    main()
