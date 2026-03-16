import hydra
from omegaconf import DictConfig
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.data.load_ultrachat import load_ultrachat_prompts
import csv
from pathlib import Path


SYSTEM_PROMPT_TEMPLATE = """
Adopt the following communication style attribute in your responses

Attribute: {attribute}

Let this attribute naturally shape your tone, word choice, and structure.
"""

@hydra.main(version_base=None, config_path="../configs", config_name="generate_responses")
def main(cfg: DictConfig):
    attributes = cfg.attributes
    num_prompts = cfg.num_prompts

    generator: LLMHelper = hydra.utils.instantiate(cfg.generator)
    generator.load()

    ultrachat_prompts = load_ultrachat_prompts(
        split="train_sft", 
        prefixes=("0", "1", "2", "3", "4", "5", "6", "7"), 
        limit=num_prompts, seed=42)
    
    metadata = []
    generator_prompts = []

    for p_idx, prompt in enumerate(ultrachat_prompts):
        for attr in attributes:
            # Positive Case
            generator_prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(attribute=attr)},
                {"role": "user", "content": prompt},
            ])
            metadata.append({"p_idx": p_idx, "attr": attr, "side": "follow", "prompt": prompt})

            # Negative Case
            generator_prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(attribute=f"not {attr}")},
                {"role": "user", "content": prompt},
            ])
            metadata.append({"p_idx": p_idx, "attr": attr, "side": "avoid", "prompt": prompt})
    
    responses = generator.generate(generator_prompts)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt_id", "attribute", "side", "prompt", "response"])
        
        for i, (meta, resp) in enumerate(zip(metadata, responses)):
            writer.writerow([
                i, 
                meta["p_idx"], 
                meta["attr"], 
                meta["side"], 
                meta["prompt"], 
                resp
            ])


if __name__ == "__main__":
    main()