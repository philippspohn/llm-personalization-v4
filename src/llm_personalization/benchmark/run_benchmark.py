from omegaconf import DictConfig
import hydra
from llm_personalization.judge.judge import AttributeJudge
from llm_personalization.benchmark.personalization_system import PersonalizationSystem, PersonalizationDataset
from llm_personalization.benchmark.attribute_benchmark.attribute_personalization_judge import PersonalizationAttributeJudge
from llm_personalization.benchmark.attribute_benchmark.attribute_personalization_dataset import AttributePersonalizationLabeledDataset, AttributePersonalizationDataset
from llm_personalization.benchmark.persona_benchmark.persona_personalization_judge import PersonaPersonalizationJudge
from llm_personalization.benchmark.persona_benchmark.persona_personalization_dataset import PersonaPersonalizationLabeledDataset, PersonaPersonalizationDataset
from llm_personalization.benchmark.robustness_benchmark.robustness_dataset import (
    load_robustness_questions, RobustnessDataset, parse_answer_letter, format_mc_prompt,
)
from pathlib import Path
from datetime import datetime
from typing import Literal
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from llm_personalization.utils.gpu_monitor import log_gpu_usage
from llm_personalization.llm.llm_helper import LLMHelper

def _generate_world_matrix(
    type: Literal["permutation", "dense"],
    shape: tuple[int, int],
    seed: int = 42,
    k: int = 1,
) -> torch.Tensor:
    """Sample a world (mapping) matrix.

    `permutation`: stacks `k` pairwise-elementwise-distinct signed permutations.
        For k=1 this is the original signed permutation (1 nonzero per row and
        per column). For k>=2 each row and each column has exactly k nonzeros
        in {-1, +1} (a k-regular signed bipartite graph). When `u` is one-hot
        at index j, `M @ u` then activates exactly `k` distinct response
        indices (those with nonzero entries in column j).

    `dense`: i.i.d. standard-normal entries (k is ignored).
    """
    rng = torch.Generator().manual_seed(seed)

    if type == "permutation":
        if shape[0] != shape[1]:
            raise ValueError(f"Permutation matrix must be square, got shape {shape}")
        n = shape[0]
        if k < 1 or k > n:
            raise ValueError(f"permutation k must satisfy 1 <= k <= n; got k={k}, n={n}")
        # Sample k permutations such that for every column they all disagree
        # (=> each column has exactly k distinct nonzero rows).
        perms: list[torch.Tensor] = []
        while len(perms) < k:
            cand = torch.randperm(n, generator=rng)
            if all(bool((cand != p).all().item()) for p in perms):
                perms.append(cand)
        matrix = torch.zeros(n, n)
        for p in perms:
            signs = torch.randint(0, 2, (n,), generator=rng) * 2 - 1
            matrix[torch.arange(n), p] = signs.float()
        return matrix

    elif type == "dense":
        return torch.randn(shape, generator=rng)

    else:
        raise ValueError(f"Unknown type: {type}")

def _user_attributes_to_vector(user_attributes: list[dict[str, str]], available_user_attributes: list[str]) -> torch.Tensor:
    vector = torch.zeros(len(available_user_attributes))
    for attribute in user_attributes:
        assert attribute["attribute"] in available_user_attributes
        assert attribute["side"] in ["follow", "avoid"]
        vector[available_user_attributes.index(attribute["attribute"])] = 1 if attribute["side"] == "follow" else -1
    return vector

def _user_attribute_vector_to_response_attribute_vector(user_attribute_vector: torch.Tensor, world_matrix: torch.Tensor) -> torch.Tensor:
    return world_matrix @ user_attribute_vector

def _response_attribute_vector_to_attributes(response_attribute_vector: torch.Tensor, available_response_attributes: list[str], max_attributes: int | None = None) -> list[dict[str, str]]:
    attributes = []
    if max_attributes is None:
        for i in range(len(available_response_attributes)):
            if response_attribute_vector[i] > 0:
                attributes.append({"attribute": available_response_attributes[i], "side": "follow"})
            elif response_attribute_vector[i] < 0:
                attributes.append({"attribute": available_response_attributes[i], "side": "avoid"})
    else:
        selected_indices = torch.argsort(response_attribute_vector.abs(), descending=True)
        for i in selected_indices[:max_attributes]:
            attributes.append({"attribute": available_response_attributes[i], "side": "follow" if response_attribute_vector[i] > 0 else "avoid"})
    return attributes

def _save_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[Benchmark]    Saved {len(records)} test samples to {path}")

def _plot_score_distribution(scores: torch.Tensor, unpersonalized_scores: torch.Tensor, output_dir: Path, world_idx: int | str):
    plt.figure(figsize=(10, 5))
    bins = np.arange(-10, 11, 1)
    plt.hist(scores.detach().cpu().numpy(), bins=bins, alpha=0.5, label="Personalized")
    plt.hist(unpersonalized_scores.detach().cpu().numpy(), bins=bins, alpha=0.5, label="Unpersonalized")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"Score Distribution (World {world_idx})")
    plt.legend()
    plt.savefig(output_dir / f"score_distribution_world_{world_idx}.png")
    plt.close()

def run_benchmark(
    benchmark_config: DictConfig,
):
    print(f"[Benchmark] Running benchmark...")
    personalization_system: PersonalizationSystem = hydra.utils.instantiate(benchmark_config.personalization_system)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(benchmark_config.output_dir.replace("{timestamp}", timestamp))
    output_dir_trained_system = output_dir / "trained_systems"
    output_dir_trained_system.mkdir(parents=True, exist_ok=True)
    output_dir_results = output_dir / "results"
    output_dir_results.mkdir(parents=True, exist_ok=True)
    num_worlds = benchmark_config.num_worlds
    world_matrix_type = benchmark_config.world_matrix_type
    world_matrix_k: int = int(benchmark_config.get("world_matrix_k", 1))
    # Optional explicit list of world seeds. If provided, overrides `num_worlds`
    # and lets you run e.g. a single seed per slurm-array task. Each seed is
    # used both as the world-matrix seed and as the world's directory name.
    world_seeds_cfg = benchmark_config.get("world_seeds", None)
    if world_seeds_cfg is not None:
        world_seeds: list[int] = [int(s) for s in world_seeds_cfg]
    else:
        world_seeds = list(range(num_worlds))
    skip_training = benchmark_config.get("skip_training", False)
    save_test_results = benchmark_config.get("save_test_results", False)
    results = {"timestamp": timestamp}

    user_attributes = list(benchmark_config.user_attributes)
    num_user_attributes_per_user = benchmark_config.num_attributes_per_user # TODO

    response_attributes = list(benchmark_config.response_attributes)
    num_response_attributes_per_user = benchmark_config.num_response_attributes_per_user
    
    # Attribute Personalization Benchmark # TODO clean up
    print(f"[Benchmark] (1/3) Attribute Personalization Benchmark")

    # Initializations
    train_labeled_dataset: AttributePersonalizationLabeledDataset = hydra.utils.instantiate(benchmark_config.attribute_personalization_dataset, split="train")
    test_labeled_dataset: AttributePersonalizationLabeledDataset = hydra.utils.instantiate(benchmark_config.attribute_personalization_dataset, split="test")
    train_dataset: AttributePersonalizationDataset = AttributePersonalizationDataset(train_labeled_dataset)
    test_dataset: AttributePersonalizationDataset = AttributePersonalizationDataset(test_labeled_dataset)
    attribute_judge: AttributeJudge = hydra.utils.instantiate(benchmark_config.attribute_judge)
    unpersonalized_llm_helper: LLMHelper = hydra.utils.instantiate(benchmark_config.unpersonalized_llm_helper)

    for world_idx in world_seeds:
        print(f"[Benchmark]    Evaluating world {world_idx}...")
        world_matrix = _generate_world_matrix(world_matrix_type, (len(user_attributes), len(response_attributes)), seed=world_idx, k=world_matrix_k)
        out_world_path = output_dir_trained_system / f"world_{world_idx}"
        out_world_path.mkdir(parents=True, exist_ok=True)

        train_id_to_attributes = {}
        test_id_to_attributes = {}
        for item in train_labeled_dataset:
            user_attribute_vector = _user_attributes_to_vector(item.user_attributes, user_attributes)
            response_attribute_vector = _user_attribute_vector_to_response_attribute_vector(user_attribute_vector, world_matrix)
            train_id_to_attributes[item.user_id] = _response_attribute_vector_to_attributes(response_attribute_vector, response_attributes, num_response_attributes_per_user)

        for item in test_labeled_dataset:
            user_attribute_vector = _user_attributes_to_vector(item.user_attributes, user_attributes)
            response_attribute_vector = _user_attribute_vector_to_response_attribute_vector(user_attribute_vector, world_matrix)
            test_id_to_attributes[item.user_id] = _response_attribute_vector_to_attributes(response_attribute_vector, response_attributes, num_response_attributes_per_user)

        personalization_attribute_judge_train = PersonalizationAttributeJudge(attribute_judge=attribute_judge, user_id_to_response_style_attributes=train_id_to_attributes)
        personalization_attribute_judge_test = PersonalizationAttributeJudge(attribute_judge=attribute_judge, user_id_to_response_style_attributes=test_id_to_attributes)

        if not skip_training:
            log_gpu_usage("Before train")
            print(f"[Benchmark]    Training personalization system...")
            personalization_system.train(train_dataset, personalization_attribute_judge_train, out_world_path, gt_user_attributes=train_id_to_attributes)
            log_gpu_usage("After train")
        else:
            print(f"[Benchmark]    Skipping training!")

        print(f"[Benchmark]    Evaluating personalization system...")
        responses = personalization_system.evaluate(test_dataset, out_world_path, gt_user_attributes=test_id_to_attributes)
        print(f"[Benchmark]    Responses generated. Avg length: {sum(len(response) for response in responses) / len(responses):.2f}")
        log_gpu_usage("After evaluate")

        test_user_ids = []
        test_conversations = []
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            test_user_ids.append(item.user_id)
            test_conversations.append(item.current_messages + [{"role": "assistant", "content": responses[i]}])


        print(f"[Benchmark]    Generating unpersoanlized responses...")
        unpersonalized_llm_helper.load()
        unpersonalized_input = [test_dataset[i].current_messages for i in range(len(test_dataset))]
        unpersonalized_responses = unpersonalized_llm_helper.generate(unpersonalized_input)
        print(f"[Benchmark]    Unpersonalized responses generated. Avg length: {sum(len(response.content) for response in unpersonalized_responses) / len(unpersonalized_responses):.2f}")
        unpersonalized_llm_helper.unload()
        log_gpu_usage("After unpersonalized responses generation")


        log_gpu_usage("Before eval judge load")
        print(f"[Benchmark]    Loading personalization attribute judge...")
        personalization_attribute_judge_test.load()
        log_gpu_usage("After eval judge load")
        print(f"[Benchmark]    Scoring responses...")
        scores = torch.tensor(personalization_attribute_judge_test.judge(test_user_ids, test_conversations))
        # print(f"[Benchmark]    Unloading personalization attribute judge...")
        # personalization_attribute_judge_test.unload()
        # log_gpu_usage("After eval judge unload")



        print(f"[Benchmark]    Scoring unpersoanlized responses...")
        # personalization_attribute_judge_test.load()
        test_conversations_unpersonalized = [
            item.current_messages + [{"role": "assistant", "content": response.content}]
            for item, response in zip(test_dataset, unpersonalized_responses)
        ]
        unpersonalized_scores = torch.tensor(personalization_attribute_judge_test.judge(test_user_ids, test_conversations_unpersonalized))
        print(f"[Benchmark]    Unloading personalization attribute judge...")
        personalization_attribute_judge_test.unload()
        log_gpu_usage("After eval judge unload")

        win_rate = (scores > unpersonalized_scores).float().mean().item()
        tie_rate = (scores == unpersonalized_scores).float().mean().item()

        print(f"[Benchmark]    Results for world {world_idx}:")
        print(f"    - Personalized system score: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"    - Unpersonalized system score: {unpersonalized_scores.mean():.4f} ± {unpersonalized_scores.std():.4f}")
        print(f"    - Win rate: {win_rate:.4f}")
        print(f"    - Tie rate: {tie_rate:.4f}")
        print(f"    - Loss rate: {1 - win_rate - tie_rate:.4f}")

        results.setdefault("attribute_benchmark", {})[f"world_{world_idx}"] = {
            "personalized_score_mean": scores.mean().item(),
            "personalized_score_std": scores.std().item(),
            "unpersonalized_score_mean": unpersonalized_scores.mean().item(),
            "unpersonalized_score_std": unpersonalized_scores.std().item(),
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "loss_rate": 1 - win_rate - tie_rate,
        }

        _plot_score_distribution(scores, unpersonalized_scores, output_dir_results, world_idx)

        if save_test_results:
            records = []
            for i in range(len(test_dataset)):
                labeled_item = test_labeled_dataset[i]
                item = test_dataset[i]
                records.append({
                    "user_id": item.user_id,
                    "conversation_history": item.conversation_history,
                    "current_messages": item.current_messages,
                    "gt_user_attributes": labeled_item.user_attributes,
                    "assigned_response_attributes": test_id_to_attributes[item.user_id],
                    "personalized_response": responses[i],
                    "unpersonalized_response": unpersonalized_responses[i].content,
                    "personalized_score": scores[i].item(),
                    "unpersonalized_score": unpersonalized_scores[i].item(),
                })
            _save_jsonl(output_dir_results / f"attribute_world_{world_idx}_test_samples.jsonl", records)
    # Persona Personalization Benchmark
    if benchmark_config.get("persona_personalization_dataset") is not None:
        print(f"[Benchmark] (2/3) Persona Personalization Benchmark")

        persona_train_labeled: PersonaPersonalizationLabeledDataset = hydra.utils.instantiate(benchmark_config.persona_personalization_dataset, split="train")
        persona_test_labeled: PersonaPersonalizationLabeledDataset = hydra.utils.instantiate(benchmark_config.persona_personalization_dataset, split="test")
        persona_train_dataset: PersonaPersonalizationDataset = PersonaPersonalizationDataset(persona_train_labeled)
        persona_test_dataset: PersonaPersonalizationDataset = PersonaPersonalizationDataset(persona_test_labeled)
        persona_judge_instance = hydra.utils.instantiate(benchmark_config.persona_judge)

        # Build user_id → formatted_persona mappings
        train_persona_map = {item.user_id: item.formatted_persona for item in persona_train_labeled}
        test_persona_map = {item.user_id: item.formatted_persona for item in persona_test_labeled}

        persona_judge_train = PersonaPersonalizationJudge(
            persona_judge=persona_judge_instance,
            user_id_to_formatted_persona=train_persona_map,
        )
        persona_judge_test = PersonaPersonalizationJudge(
            persona_judge=persona_judge_instance,
            user_id_to_formatted_persona=test_persona_map,
        )

        persona_system: PersonalizationSystem = hydra.utils.instantiate(benchmark_config.persona_personalization_system)
        out_persona_path = output_dir_trained_system / "persona"
        out_persona_path.mkdir(parents=True, exist_ok=True)

        if not skip_training:
            log_gpu_usage("Before persona train")
            print(f"[Benchmark]    Training persona personalization system...")
            persona_system.train(persona_train_dataset, persona_judge_train, out_persona_path)
            log_gpu_usage("After persona train")
        else:
            print(f"[Benchmark]    Skipping persona training!")

        print(f"[Benchmark]    Evaluating persona personalization system...")
        persona_responses = persona_system.evaluate(persona_test_dataset, out_persona_path)
        print(f"[Benchmark]    Responses generated. Avg length: {sum(len(r) for r in persona_responses) / len(persona_responses):.2f}")
        log_gpu_usage("After persona evaluate")

        persona_test_user_ids = [persona_test_dataset[i].user_id for i in range(len(persona_test_dataset))]
        persona_test_conversations = [
            persona_test_dataset[i].current_messages + [{"role": "assistant", "content": persona_responses[i]}]
            for i in range(len(persona_test_dataset))
        ]

        print(f"[Benchmark]    Generating unpersonalized responses...")
        unpersonalized_llm_helper.load()
        unpersonalized_input = [persona_test_dataset[i].current_messages for i in range(len(persona_test_dataset))]
        unpersonalized_persona_responses = unpersonalized_llm_helper.generate(unpersonalized_input)
        print(f"[Benchmark]    Unpersonalized responses generated. Avg length: {sum(len(r.content) for r in unpersonalized_persona_responses) / len(unpersonalized_persona_responses):.2f}")
        unpersonalized_llm_helper.unload()
        log_gpu_usage("After persona unpersonalized responses")

        persona_test_conversations_unpersonalized = [
            persona_test_dataset[i].current_messages + [{"role": "assistant", "content": resp.content}]
            for i, resp in enumerate(unpersonalized_persona_responses)
        ]

        log_gpu_usage("Before persona judge load")
        persona_judge_test.load()
        log_gpu_usage("After persona judge load")
        print(f"[Benchmark]    Scoring personalized responses...")
        persona_scores = torch.tensor(persona_judge_test.judge(persona_test_user_ids, persona_test_conversations), dtype=torch.float)
        print(f"[Benchmark]    Scoring unpersonalized responses...")
        persona_unpersonalized_scores = torch.tensor(persona_judge_test.judge(persona_test_user_ids, persona_test_conversations_unpersonalized), dtype=torch.float)
        persona_judge_test.unload()
        log_gpu_usage("After persona judge unload")

        persona_win_rate = (persona_scores > persona_unpersonalized_scores).float().mean().item()
        persona_tie_rate = (persona_scores == persona_unpersonalized_scores).float().mean().item()

        print(f"[Benchmark]    Persona Benchmark Results:")
        print(f"    - Personalized system score: {persona_scores.mean():.4f} ± {persona_scores.std():.4f}")
        print(f"    - Unpersonalized system score: {persona_unpersonalized_scores.mean():.4f} ± {persona_unpersonalized_scores.std():.4f}")
        print(f"    - Win rate: {persona_win_rate:.4f}")
        print(f"    - Tie rate: {persona_tie_rate:.4f}")
        print(f"    - Loss rate: {1 - persona_win_rate - persona_tie_rate:.4f}")

        results["persona_benchmark"] = {
            "personalized_score_mean": persona_scores.mean().item(),
            "personalized_score_std": persona_scores.std().item(),
            "unpersonalized_score_mean": persona_unpersonalized_scores.mean().item(),
            "unpersonalized_score_std": persona_unpersonalized_scores.std().item(),
            "win_rate": persona_win_rate,
            "tie_rate": persona_tie_rate,
            "loss_rate": 1 - persona_win_rate - persona_tie_rate,
        }

        _plot_score_distribution(persona_scores, persona_unpersonalized_scores, output_dir_results, world_idx="persona")

        if save_test_results:
            records = []
            for i in range(len(persona_test_dataset)):
                labeled_item = persona_test_labeled[i]
                item = persona_test_dataset[i]
                records.append({
                    "user_id": item.user_id,
                    "conversation_history": item.conversation_history,
                    "current_messages": item.current_messages,
                    "persona": labeled_item.persona,
                    "formatted_persona": labeled_item.formatted_persona,
                    "personalized_response": persona_responses[i],
                    "unpersonalized_response": unpersonalized_persona_responses[i].content,
                    "personalized_score": persona_scores[i].item(),
                    "unpersonalized_score": persona_unpersonalized_scores[i].item(),
                })
            _save_jsonl(output_dir_results / "persona_test_samples.jsonl", records)

    # Robustness Benchmark
    robustness_config = benchmark_config.get("robustness")
    if robustness_config is not None:
        print(f"[Benchmark] (3/3) Robustness Benchmark")

        robustness_questions = load_robustness_questions(
            include_mmlu_pro=robustness_config.get("include_mmlu_pro", True),
            include_truthfulqa=robustness_config.get("include_truthfulqa", True),
            include_bbq=robustness_config.get("include_bbq", False),
            mmlu_pro_limit=robustness_config.get("mmlu_pro_limit", None),
            truthfulqa_limit=robustness_config.get("truthfulqa_limit", None),
            bbq_limit=robustness_config.get("bbq_limit", None),
            seed=benchmark_config.get("seed", 42),
        )
        print(f"[Benchmark]    Loaded {len(robustness_questions)} robustness questions")

        # Build robustness datasets from the existing test datasets
        robustness_systems: list[tuple[str, PersonalizationSystem, Path, PersonalizationDataset]] = []

        # Attribute personalization system (use last world)
        if test_dataset is not None:
            attr_robustness_dataset = RobustnessDataset(test_dataset, robustness_questions)
            last_world_path = output_dir_trained_system / f"world_{world_seeds[-1]}"
            robustness_systems.append(("attribute", personalization_system, last_world_path, attr_robustness_dataset))

        # Persona personalization system
        if benchmark_config.get("persona_personalization_dataset") is not None:
            persona_robustness_dataset = RobustnessDataset(persona_test_dataset, robustness_questions)
            robustness_systems.append(("persona", persona_system, out_persona_path, persona_robustness_dataset))

        # Unpersonalized baseline: just the questions (no user history)
        print(f"[Benchmark]    Generating unpersonalized robustness responses...")
        unpersonalized_llm_helper.load()
        unpersonalized_prompts = [
            [{"role": "user", "content": format_mc_prompt(q)}]
            for q in robustness_questions
        ]
        unpersonalized_robustness_responses = unpersonalized_llm_helper.generate(unpersonalized_prompts)
        unpersonalized_llm_helper.unload()
        log_gpu_usage("After unpersonalized robustness responses")

        unpersonalized_correct = {"mmlu_pro": 0, "truthfulqa": 0, "bbq": 0}
        unpersonalized_total = {"mmlu_pro": 0, "truthfulqa": 0, "bbq": 0}
        for q, resp in zip(robustness_questions, unpersonalized_robustness_responses):
            parsed = parse_answer_letter(resp.content)
            unpersonalized_total[q.source] += 1
            if parsed == q.correct_letter:
                unpersonalized_correct[q.source] += 1

        if save_test_results:
            records = []
            for q, resp in zip(robustness_questions, unpersonalized_robustness_responses):
                records.append({
                    "question_id": q.question_id,
                    "source": q.source,
                    "question_text": q.question_text,
                    "options": q.options,
                    "option_letters": q.option_letters,
                    "correct_letter": q.correct_letter,
                    "response": resp.content,
                    "parsed_letter": parse_answer_letter(resp.content),
                })
            _save_jsonl(output_dir_results / "robustness_unpersonalized_test_samples.jsonl", records)

        robustness_results = {"unpersonalized": {}}
        print(f"[Benchmark]    Unpersonalized robustness results:")
        for source in ["mmlu_pro", "truthfulqa", "bbq"]:
            if unpersonalized_total[source] > 0:
                acc = unpersonalized_correct[source] / unpersonalized_total[source]
                print(f"      {source}: {unpersonalized_correct[source]}/{unpersonalized_total[source]} = {acc:.4f}")
                robustness_results["unpersonalized"][source] = {
                    "correct": unpersonalized_correct[source],
                    "total": unpersonalized_total[source],
                    "accuracy": acc,
                }

        # Evaluate each personalized system
        for system_name, system, load_path, robustness_dataset in robustness_systems:
            print(f"[Benchmark]    Evaluating robustness for {system_name} system ({len(robustness_dataset)} items)...")
            log_gpu_usage(f"Before {system_name} robustness evaluate")
            robustness_responses = system.evaluate(robustness_dataset, load_path)
            log_gpu_usage(f"After {system_name} robustness evaluate")

            correct = {"mmlu_pro": 0, "truthfulqa": 0, "bbq": 0}
            total = {"mmlu_pro": 0, "truthfulqa": 0, "bbq": 0}
            for i, resp in enumerate(robustness_responses):
                q = robustness_dataset.get_question(i)
                parsed = parse_answer_letter(resp)
                total[q.source] += 1
                if parsed == q.correct_letter:
                    correct[q.source] += 1

            if save_test_results:
                records = []
                for i, resp in enumerate(robustness_responses):
                    q = robustness_dataset.get_question(i)
                    item = robustness_dataset[i]
                    records.append({
                        "user_id": item.user_id,
                        "conversation_history": item.conversation_history,
                        "question_id": q.question_id,
                        "source": q.source,
                        "question_text": q.question_text,
                        "options": q.options,
                        "option_letters": q.option_letters,
                        "correct_letter": q.correct_letter,
                        "response": resp,
                        "parsed_letter": parse_answer_letter(resp),
                    })
                _save_jsonl(output_dir_results / f"robustness_{system_name}_test_samples.jsonl", records)

            robustness_results[system_name] = {}
            print(f"[Benchmark]    Robustness results for {system_name} system:")
            for source in ["mmlu_pro", "truthfulqa", "bbq"]:
                if total[source] > 0:
                    acc = correct[source] / total[source]
                    baseline_acc = unpersonalized_correct[source] / unpersonalized_total[source] if unpersonalized_total[source] > 0 else 0
                    delta = acc - baseline_acc
                    print(f"      {source}: {correct[source]}/{total[source]} = {acc:.4f} (delta vs unpersonalized: {delta:+.4f})")
                    robustness_results[system_name][source] = {
                        "correct": correct[source],
                        "total": total[source],
                        "accuracy": acc,
                        "delta_vs_unpersonalized": delta,
                    }

        results["robustness_benchmark"] = robustness_results

    # Save all results
    results_path = output_dir_results / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Benchmark] Results saved to {results_path}")

