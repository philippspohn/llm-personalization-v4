from llm_personalization.benchmark.run_benchmark import run_benchmark
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run_benchmark(cfg)

if __name__ == "__main__":
    main()