from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from llm_personalization.judge.judge import AttributeJudge
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from .cache import CachedDataset
from .judging import WeightedAttributeJudge
from .methods.base import PersonalizationSystemV3
from .preference_mapping import (
    generate_world_matrix,
    map_to_response_vector,
    response_vector_to_weighted_attributes,
    user_attributes_to_vector,
)
from .robustness import run_robustness, unpers_cache_key
from .unpersonalized_cache import UnpersonalizedCache


def _gt_for_world(
    dataset: CachedDataset,
    world_matrix: torch.Tensor,
    user_attributes: list[str],
    response_attributes: list[str],
    max_response_attributes: int | None,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for user in dataset:
        u_vec = user_attributes_to_vector(user.gt_user_attributes, user_attributes)
        r_vec = map_to_response_vector(u_vec, world_matrix)
        out[user.user_id] = response_vector_to_weighted_attributes(
            r_vec, response_attributes, max_response_attributes
        )
    return out


def _save_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[BenchmarkV3] saved {len(records)} records to {path}")


def _plot_score_distribution(scores: torch.Tensor, baseline: torch.Tensor, output_dir: Path, world_idx) -> None:
    plt.figure(figsize=(10, 5))
    bins = np.arange(-10, 11, 1)
    plt.hist(scores.detach().cpu().numpy(), bins=bins, alpha=0.5, label="Personalized")
    plt.hist(baseline.detach().cpu().numpy(), bins=bins, alpha=0.5, label="Unpersonalized")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"Score Distribution (World {world_idx})")
    plt.legend()
    plt.savefig(output_dir / f"score_distribution_world_{world_idx}.png")
    plt.close()


def run_benchmark(cfg: DictConfig) -> None:
    print("[BenchmarkV3] Running benchmark...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Append a tag derived from world_seeds (or num_worlds) so concurrent jobs
    # that happen to start in the same second don't collide on output_dir.
    world_seeds_for_tag = cfg.get("world_seeds", None)
    if world_seeds_for_tag is not None:
        seed_tag = "seeds" + "_".join(str(int(s)) for s in world_seeds_for_tag)
    else:
        seed_tag = f"w{int(cfg.num_worlds)}"
    output_dir = Path(str(cfg.output_dir).replace("{timestamp}", f"{timestamp}_{seed_tag}"))
    out_systems = output_dir / "trained_systems"
    out_results = output_dir / "results"
    out_systems.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    user_attributes = list(cfg.user_attributes)
    response_attributes = list(cfg.response_attributes)
    max_response_attributes = cfg.get("num_response_attributes_per_user", None)
    world_matrix_type = cfg.world_matrix_type
    world_matrix_k = int(cfg.get("world_matrix_k", 1))

    world_seeds_cfg = cfg.get("world_seeds", None)
    if world_seeds_cfg is not None:
        world_seeds = [int(s) for s in world_seeds_cfg]
    else:
        world_seeds = list(range(int(cfg.num_worlds)))

    skip_training = cfg.get("skip_training", False)
    save_test_results = cfg.get("save_test_results", True)

    # Load datasets once (shared across worlds)
    train_dataset = CachedDataset(
        split="train",
        conversations_path=cfg.cache.conversations_train,
        responses_path=cfg.cache.responses_train,
        ratings_path=cfg.cache.ratings_train,
        history_max_len=cfg.cache.get("history_max_len", None),
        limit=cfg.cache.get("train_limit", None),
    )
    val_size = int(cfg.cache.get("val_size", 0) or 0)
    val_dataset = train_dataset.split_off_val(val_size) if val_size > 0 else None
    test_dataset = CachedDataset(
        split="test",
        conversations_path=cfg.cache.conversations_test,
        responses_path=cfg.cache.responses_test,
        ratings_path=cfg.cache.ratings_test,
        history_max_len=cfg.cache.get("history_max_len", None),
        limit=cfg.cache.get("test_limit", None),
    )

    attribute_judge: AttributeJudge = hydra.utils.instantiate(cfg.attribute_judge)
    unpersonalized_cache = UnpersonalizedCache(
        responses_path=cfg.cache.unpersonalized_responses_test,
        ratings_path=cfg.cache.unpersonalized_ratings_test,
    )

    results: dict = {"timestamp": timestamp, "config": {"method": cfg.method_name}}

    for world_idx in world_seeds:
        print(f"[BenchmarkV3] === world {world_idx} ===")
        world_matrix = generate_world_matrix(
            world_matrix_type,
            (len(user_attributes), len(response_attributes)),
            seed=world_idx,
            k=world_matrix_k,
        )
        train_gt = _gt_for_world(
            train_dataset, world_matrix, user_attributes, response_attributes, max_response_attributes
        )
        val_gt = (
            _gt_for_world(val_dataset, world_matrix, user_attributes, response_attributes, max_response_attributes)
            if val_dataset is not None else None
        )
        test_gt = _gt_for_world(
            test_dataset, world_matrix, user_attributes, response_attributes, max_response_attributes
        )

        system: PersonalizationSystemV3 = hydra.utils.instantiate(cfg.personalization_system)
        world_path = out_systems / f"world_{world_idx}"
        world_path.mkdir(parents=True, exist_ok=True)

        if not skip_training:
            log_gpu_usage("Before train")
            system.train(
                train_dataset, train_gt, world_path,
                val_dataset=val_dataset, val_user_id_to_weighted_attrs=val_gt,
            )
            log_gpu_usage("After train")
        else:
            print("[BenchmarkV3] skipping training")

        # ---- generate personalized responses (single batched call) ----
        print("[BenchmarkV3] generating personalized responses...")
        responses = system.evaluate(test_dataset, world_path, user_id_to_weighted_attrs=test_gt)

        test_user_ids = [u.user_id for u in test_dataset]
        test_conversations = [
            u.current_messages + [{"role": "assistant", "content": r}]
            for u, r in zip(test_dataset, responses)
        ]

        # ---- unpersonalized baseline: pure cache lookup (no LLM/judge calls) ----
        print("[BenchmarkV3] looking up unpersonalized responses + scores from cache...")
        missing = sum(1 for uid in test_user_ids if uid not in unpersonalized_cache)
        if missing:
            print(f"[BenchmarkV3] WARNING: {missing}/{len(test_user_ids)} users missing from unpers cache")
        unpers_responses_text: list[str] = [
            (unpersonalized_cache.get(uid).response if uid in unpersonalized_cache else "")
            for uid in test_user_ids
        ]
        unpers_scores = torch.tensor(
            [unpersonalized_cache.weighted_score(uid, test_gt[uid]) for uid in test_user_ids],
            dtype=torch.float,
        )

        # ---- score personalized with the live judge ----
        wj = WeightedAttributeJudge(attribute_judge, test_gt)
        log_gpu_usage("Before judge load")
        wj.load()
        log_gpu_usage("After judge load")
        print("[BenchmarkV3] scoring personalized...")
        scores = torch.tensor(wj.judge(test_user_ids, test_conversations), dtype=torch.float)
        wj.unload()
        log_gpu_usage("After judge unload")

        valid = ~(torch.isnan(scores) | torch.isnan(unpers_scores))
        win_rate = (scores[valid] > unpers_scores[valid]).float().mean().item() if valid.any() else float("nan")
        tie_rate = (scores[valid] == unpers_scores[valid]).float().mean().item() if valid.any() else float("nan")

        print(f"[BenchmarkV3] World {world_idx} results:")
        print(f"  - Personalized:   {scores[valid].mean():.4f} +/- {scores[valid].std():.4f}")
        print(f"  - Unpersonalized: {unpers_scores[valid].mean():.4f} +/- {unpers_scores[valid].std():.4f}")
        print(f"  - Win/Tie/Loss:   {win_rate:.4f} / {tie_rate:.4f} / {1 - win_rate - tie_rate:.4f}")

        results.setdefault("attribute_benchmark", {})[f"world_{world_idx}"] = {
            "personalized_score_mean": float(scores[valid].mean()),
            "personalized_score_std": float(scores[valid].std()),
            "unpersonalized_score_mean": float(unpers_scores[valid].mean()),
            "unpersonalized_score_std": float(unpers_scores[valid].std()),
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "loss_rate": 1 - win_rate - tie_rate,
            "n_valid": int(valid.sum()),
            "n_total": len(scores),
        }

        _plot_score_distribution(scores[valid], unpers_scores[valid], out_results, world_idx)

        if save_test_results:
            records = []
            for i, user in enumerate(test_dataset):
                records.append({
                    "user_id": user.user_id,
                    "current_messages": user.current_messages,
                    "gt_user_attributes": user.gt_user_attributes,
                    "weighted_response_targets": test_gt[user.user_id],
                    "personalized_response": responses[i],
                    "unpersonalized_response": unpers_responses_text[i],
                    "personalized_score": float(scores[i]),
                    "unpersonalized_score": float(unpers_scores[i]),
                })
            _save_jsonl(out_results / f"attribute_world_{world_idx}_test_samples.jsonl", records)

        # ---- robustness (gated by config presence; submitter adds for one job only) ----
        robustness_cfg = cfg.get("robustness", None)
        if robustness_cfg is not None:
            unpers_cache_path = (
                Path("data/unpersonalized_robustness")
                / str(cfg.gen_model_tag)
                / f"{unpers_cache_key(robustness_cfg)}.jsonl"
            )
            print(f"[BenchmarkV3] running robustness eval for world {world_idx}...")
            rob_result = run_robustness(
                system=system,
                test_dataset=test_dataset,
                world_path=world_path,
                world_idx=world_idx,
                robustness_cfg=robustness_cfg,
                unpers_cache_path=unpers_cache_path,
                output_dir=out_results,
                save_test_results=save_test_results,
                user_id_to_weighted_attrs=test_gt,
            )
            results.setdefault("robustness_benchmark", {})[f"world_{world_idx}"] = rob_result

    results_path = out_results / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[BenchmarkV3] results saved to {results_path}")
