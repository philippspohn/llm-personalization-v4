import hydra
from omegaconf import DictConfig
import torch


TEST_PROMPT = "What is the capital of France?"
TEST_RESPONSE_1 = """
While you might be looking for a simple one-word answer, the capital of **France** is a city that carries enough historical, political, and cultural weight to deserve a bit more "elaborate" treatment.

## The Heart of France: Paris
**Paris** is the undisputed capital and the largest city of France. Situated on a northward-bending arc of the river **Seine**, it has been the center of French power for over a thousand years.

---

### Why Paris Matters
To understand Paris as a capital, it helps to look at it through three specific lenses:

* **Political Centralization:** Unlike many countries where power is spread out, France is highly centralized. Paris is the seat of the national government, home to the **Élysée Palace** (the official residence of the President) and the **Hôtel Matignon** (the residence of the Prime Minister).
* **Historical Legacy:** Originally a Gallic settlement called *Lutetia*, it became a major center of Western Christendom and education in the 12th century. It played a starring role in the **French Revolution**, which fundamentally changed the course of modern history and democratic ideals.
* **Cultural & Economic Hub:** Often called the "City of Light" (*La Ville Lumière*), Paris is a global leader in fashion, gastronomy, and the arts. It houses the **Louvre**, the world's most-visited art museum, and iconic landmarks like the **Eiffel Tower** and **Notre-Dame Cathedral**.

### Geographical Layout
The city is famously divided by the Seine into two distinct areas:
1.  **The Rive Droite (Right Bank):** Known historically for commerce, trade, and the grand boulevards designed by Georges-Eugène Haussmann.
2.  **The Rive Gauche (Left Bank):** Traditionally the bohemian and intellectual heart of the city, home to the Latin Quarter and the Sorbonne University.

---

### Fun Fact
Paris wasn't *always* the capital. During various periods of war or political upheaval—most notably during World War II—the seat of government was temporarily moved to cities like **Tours**, **Bordeaux**, and **Vichy**. However, the soul of the French state has always returned to the banks of the Seine.

Would you like me to dive deeper into the history of how Paris became the capital, or perhaps help you plan a virtual itinerary of its most famous landmarks?
"""
TEST_RESPONSE_2 = "The capital of France is Paris. It is also the largest city in France."
TEST_RESPONSE_3 = "Paris"
TEST_PRINCIPLE = "conciseness"


def print_gpu_info() -> None:
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name}  ({vram_gb:.1f} GB)")
    print("=" * 50)


def test_principle_judge(cfg: DictConfig) -> None:
    from llm_personalization.judge import PrincipleJudge

    judge = PrincipleJudge(
        model=cfg.judge.model,
        torch_dtype=getattr(torch, cfg.judge.torch_dtype),
        device_map=cfg.judge.device_map,
    )
    judge.load_model()

    scores = judge.judge(
        prompts=[TEST_PROMPT, TEST_PROMPT, TEST_PROMPT],
        responses=[TEST_RESPONSE_1, TEST_RESPONSE_2, TEST_RESPONSE_3],
        principles=[TEST_PRINCIPLE, TEST_PRINCIPLE, TEST_PRINCIPLE],
        batch_size=cfg.judge.batch_size,
    )
    print(f"PrincipleJudge scores (P(Yes)): {scores[0]:.4f}, {scores[1]:.4f}, {scores[2]:.4f}")
    judge.unload_model()


def test_rating_judge(cfg: DictConfig) -> None:
    from llm_personalization.judge import RatingJudge

    judge = RatingJudge(
        model=cfg.judge.model,
        tensor_parallel_size=cfg.judge.tensor_parallel_size,
        gpu_memory_utilization=cfg.judge.gpu_memory_utilization,
        range=(0, 9),
    )
    judge.load_llm()

    SYSTEM_PROMPT = "You are a helpful assistant. Rate the response quality on a scale of 0-9."
    scores = judge.judge(
        prompts=[TEST_PROMPT, TEST_PROMPT, TEST_PROMPT],
        responses=[TEST_RESPONSE_1, TEST_RESPONSE_2, TEST_RESPONSE_3],
        evaluation_system_prompts=[SYSTEM_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT],
    )
    # print(f"RatingJudge scores (0-10): {scores[0]:.4f}, {scores[1]:.4f}, {scores[2]:.4f}")
    print(f"RatingJudge scores (0-9): {scores[0]:.4f}, {scores[1]:.4f}, {scores[2]:.4f}")
    judge.unload_llm()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print_gpu_info()

    print(f"\nTesting judge type: {cfg.judge.type}")
    print(f"Model: {cfg.judge.model}\n")

    if cfg.judge.type == "principle":
        test_principle_judge(cfg)
    elif cfg.judge.type == "rating":
        test_rating_judge(cfg)
    else:
        raise ValueError(f"Unknown judge type: {cfg.judge.type}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
