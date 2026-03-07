import hydra
from omegaconf import DictConfig
from llm_personalization.llm.llm_helper import LLMHelper
from hydra.utils import instantiate
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    conversations = load_ultrachat_conversations(
        split="train_sft",
        limit=cfg.limit,
        seed=42,
    )

    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()


    llm.unload()

if __name__ == "__main__":
    main()