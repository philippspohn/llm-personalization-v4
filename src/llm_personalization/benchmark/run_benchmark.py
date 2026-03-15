from omegaconf import DictConfig
import hydra
from llm_personalization.judge.judge import AttributeJudge
from llm_personalization.benchmark.personalization_system import PersonalizationSystem, PersonalizationDataset
from llm_personalization.benchmark.attribute_benchmark.attribute_personalization_judge import PersonalizationAttributeJudge
from llm_personalization.benchmark.attribute_benchmark.attribute_personalization_dataset import AttributePersonalizationLabeledDataset, AttributePersonalizationDataset
from pathlib import Path
from datetime import datetime
from typing import Literal
import torch
import random
from llm_personalization.utils.gpu_monitor import log_gpu_usage
from llm_personalization.llm.llm_helper import LLMHelper

def _generate_world_matrix(type: Literal["permutation", "dense"], shape: tuple[int, int], seed: int = 42) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)

    if type == "permutation":
        if shape[0] != shape[1]:
            raise ValueError(f"Permutation matrix must be square, got shape {shape}")
        n = shape[0]
        perm = torch.randperm(n, generator=rng)
        matrix = torch.zeros(n, n)
        matrix[torch.arange(n), perm] = 1.0
        # Randomly flip signs
        signs = torch.randint(0, 2, (n,), generator=rng) * 2 - 1  # -1 or +1
        matrix[torch.arange(n), perm] = signs.float()
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
    skip_training = benchmark_config.get("skip_training", False)
    
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

    for world_idx in range(num_worlds):
        print(f"[Benchmark]    Evaluating world {world_idx}...")
        world_matrix = _generate_world_matrix(world_matrix_type, (len(user_attributes), len(response_attributes)), seed=world_idx)
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
            personalization_system.train(train_dataset, personalization_attribute_judge_train, out_world_path)
            log_gpu_usage("After train")
        else:
            print(f"[Benchmark]    Skipping training!")

        print(f"[Benchmark]    Evaluating personalization system...")
        responses = personalization_system.evaluate(test_dataset, out_world_path)
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
        unpersonalized_responses = unpersonalized_llm_helper.generate(test_conversations)
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
    # Persona Personalization Benchmark

    # TODO

    # Robustness Benchmark


    
