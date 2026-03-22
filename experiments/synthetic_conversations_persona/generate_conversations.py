import hydra
from omegaconf import DictConfig
from llm_personalization.llm.llm_helper import LLMHelper
from hydra.utils import instantiate, get_original_cwd
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations_with_ids
from llm_personalization.data.load_nemotron_personas import load_nemotron_personas

from pathlib import Path
import json
from tqdm import tqdm

SYSTEM_PROMPT = """
Your task: Rewrite the highlighted user prompt to match a described persona. Output only the rewritten prompt, nothing else.

Rules:
- Preserve the core meaning, intent, and information content of the original prompt.
- Only rewrite and output the "prompt_to_rewrite" message, the preceding conversation is just for context.
- Change the way it is expressed — word choice, sentence structure, tone, and register — to match the target style. Apply the style NATURALLY — the rewrite should still sound like a real person, not a caricature. Less is more!
- Do not mention any attributes of the persona explicitly in the rewritten prompt.
- Keep roughly the same length unless the style naturally calls for shorter or longer phrasing.
"""

USER_PROMPT_TEMPLATE = """
Rewrite the highlighted user prompt to match a described persona. Output only the rewritten prompt, nothing else.

<conversation>
{conversation}
</conversation>

<prompt_to_rewrite>
{prompt_to_rewrite}
</prompt_to_rewrite>

<persona>
{formatted_persona}
</persona>
"""

def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n".join([f"<message role='{message['role']}'>{message['content']}</message>" for message in conversation])

def _format_persona(persona: dict[str, str]) -> str:
    parts = [
        f"PERSONA:\n{persona['persona']}",
        f"SKILLS AND EXPERTISE:\n{persona['skills_and_expertise']}",
        f"HOBBIES AND INTERESTS:\n{persona['hobbies_and_interests']}",
    ]
    return "\n\n".join(parts)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    oversample_factor = 1.05 # Over-sample to account for failed requests
    num_train_users = cfg.num_train_users
    num_sampled_train_users = int(num_train_users * oversample_factor)
    num_test_users = cfg.num_test_users
    num_sampled_test_users = int(num_test_users * oversample_factor)
    num_conversations_per_user = cfg.num_conversations_per_user
    seed = cfg.get("seed", 42)

    train_conversations_with_ids = load_ultrachat_conversations_with_ids(
        split="train_sft",
        seed=seed,
    )
    test_conversations_with_ids = load_ultrachat_conversations_with_ids(
        split="test_sft",
        seed=seed,
    )

    # Load personas (one per user)
    print("Loading personas...")
    train_personas = load_nemotron_personas(
        split="train", limit=num_sampled_train_users, seed=seed,
    )
    test_personas = load_nemotron_personas(
        split="test", limit=num_sampled_test_users, seed=seed,
    )

    first_message_only = cfg.get("first_message_only", False)

    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()

    # Build user structures and generation requests.
    # Each user has a list of conversations, each conversation has a list of
    # (request_index, turn_index) pairs so we can map responses back.
    generation_requests = []
    users = []  # list of user dicts
    global_conv_idx = 0
    for split in ["train", "test"]:
        conversations_with_ids = train_conversations_with_ids if split == "train" else test_conversations_with_ids
        personas = train_personas if split == "train" else test_personas
        num_sampled_users = num_sampled_train_users if split == "train" else num_sampled_test_users
        num_sampled_users = min(num_sampled_users, len(personas))
        print(f"[{split}] Generating {num_sampled_users} users")
        for usr_idx in tqdm(range(num_sampled_users), desc=f"[{split}] Generating users"):
            persona = personas[usr_idx]
            formatted_persona = _format_persona(persona)
            user = {
                "split": split,
                "user_idx": usr_idx,
                "persona": persona,
                "conversations": [],  # list of (conversation_id, original_conversation, request_indices)
            }
            for _ in range(num_conversations_per_user):
                conversation_id, conversation = conversations_with_ids[global_conv_idx % len(conversations_with_ids)]
                global_conv_idx += 1

                # Each user turn (even indices: 0, 2, 4, ...) gets a rewrite request
                # If first_message_only, only rewrite the first user turn (index 0)
                turn_indices = [0] if first_message_only else range(0, len(conversation), 2)
                request_indices = []  # (request_idx, turn_index_in_conversation)
                for i in turn_indices:
                    preceding_context = conversation[:i]
                    request_idx = len(generation_requests)
                    generation_requests.append([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(
                                conversation=_format_conversation(preceding_context),
                                prompt_to_rewrite=conversation[i]["content"],
                                formatted_persona=formatted_persona,
                            ),
                        },
                    ])
                    request_indices.append((request_idx, i))
                user["conversations"].append((conversation_id, conversation, request_indices))
            users.append(user)

    print(f"Generating {len(generation_requests)} requests...", flush=True)
    model_responses = llm.generate(generation_requests)
    print(f"Generated {len(model_responses)} responses", flush=True)

    # Validate responses, reconstruct conversations, and group by user
    output_users = {"train": [], "test": []}
    num_discarded = {"train": 0, "test": 0}
    for user in tqdm(users, desc="Validating responses"):
        split = user["split"]

        # Check that all generation requests for this user succeeded
        all_request_indices = [
            req_idx
            for _, _, request_indices in user["conversations"]
            for req_idx, _ in request_indices
        ]
        if any(
            not model_responses[idx].content or not model_responses[idx].finish_reason_stop
            for idx in all_request_indices
        ):
            num_discarded[split] += 1
            continue

        # Reconstruct conversations with rewritten user prompts
        rewritten_conversations = []
        for conversation_id, original_conversation, request_indices in user["conversations"]:
            rewritten = [msg.copy() for msg in original_conversation]
            for req_idx, turn_idx in request_indices:
                rewritten[turn_idx] = {
                    "role": "user",
                    "content": model_responses[req_idx].content,
                }
            # If first_message_only, keep only the first user prompt + assistant response
            if first_message_only:
                rewritten = rewritten[:2]
            rewritten_conversations.append({
                "conversation_id": conversation_id,
                "messages": rewritten,
            })
        output_users[split].append({
            "user_idx": user["user_idx"],
            "persona_uuid": user["persona"]["uuid"],
            "persona": user["persona"],
            "formatted_persona": _format_persona(user["persona"]),
            "conversations": rewritten_conversations,
        })

    # Trim to the requested number of users and save
    for split, num_target, path_key in tqdm([("train", num_train_users, "train_output_path"), ("test", num_test_users, "test_output_path")], desc="Saving users"):
        split_users = output_users[split][:num_target]
        print(f"[{split}] Kept {len(split_users)}/{num_target} users, "
              f"discarded {num_discarded[split]} due to failed generations")

        out_path = Path(get_original_cwd()) / cfg[path_key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for user_data in split_users:
                f.write(json.dumps(user_data) + "\n")
        print(f"[{split}] Saved to {out_path}")

    llm.unload()

if __name__ == "__main__":
    main()
