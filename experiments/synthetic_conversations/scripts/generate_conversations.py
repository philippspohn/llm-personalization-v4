import hydra
from omegaconf import DictConfig
from llm_personalization.llm.llm_helper import LLMHelper
from hydra.utils import instantiate, get_original_cwd
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations_with_ids

from pathlib import Path
import random
import json
from tqdm import tqdm

SYSTEM_PROMPT = """
Your task: Rewrite the highlighted user prompt to match a described persona. Output only the rewritten prompt, nothing else.

Rules:
- Preserve the core meaning, intent, and information content of the original prompt.
- Only rewrite and output the "prompt_to_rewrite" message, the preceding conversation is just for context.
- Change the way it is expressed — word choice, sentence structure, tone, and register — to match the target style. Apply the style NATURALLY — the rewrite should still sound like a real person, not a caricature. Less is more!
- Do not mention any style attribute explicitly in the rewritten prompt.
- Keep roughly the same length unless the style naturally calls for shorter or longer phrasing (e.g., a terse style should be shorter, a verbose style should be longer).
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
The simulated user is...
{formatted_attributes}
</persona>
"""

def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n".join([f"<message role='{message['role']}'>{message['content']}</message>" for message in conversation])

def _format_rewrite_style(attributes: list[dict[str, str]]) -> str:
    follow = [a["attribute"] for a in attributes if a["side"] == "follow"]
    avoid = [a["attribute"] for a in attributes if a["side"] == "avoid"]
    parts = []
    if follow:
        parts.append("VERY: " + ", ".join(follow))
    if avoid:
        parts.append("NOT AT ALL: " + ", ".join(avoid))
    return "\n".join(parts)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    oversample_factor = 1.05 # Over-sample to account for failed requests
    num_train_users = cfg.num_train_users
    num_sampled_train_users = int(num_train_users * oversample_factor) 
    num_test_users = cfg.num_test_users
    num_sampled_test_users = int(num_test_users * oversample_factor) 
    num_conversations_per_user = cfg.num_conversations_per_user
    num_attributes_per_user = cfg.num_attributes_per_user
    attributes = list(cfg.attributes)
    seed = cfg.get("seed", 42)

    train_conversations_with_ids = load_ultrachat_conversations_with_ids(
        split="train_sft",
        seed=seed,
    )
    test_conversations_with_ids = load_ultrachat_conversations_with_ids(
        split="test_sft",
        seed=seed,
    )

    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()

    # Build user structures and generation requests.
    # Each user has a list of conversations, each conversation has a list of
    # (request_index, turn_index) pairs so we can map responses back.
    generation_requests = []
    users = []  # list of user dicts
    rng = random.Random(seed)
    global_conv_idx = 0
    for split in ["train", "test"]:
        conversations_with_ids = train_conversations_with_ids if split == "train" else test_conversations_with_ids
        num_sampled_users = num_sampled_train_users if split == "train" else num_sampled_test_users
        print(f"[{split}] Generating {num_sampled_users} users")
        for usr_idx in tqdm(range(num_sampled_users), desc=f"[{split}] Generating users"):
            sampled_attrs = rng.sample(attributes, num_attributes_per_user)
            rewrite_style_attributes = [
                {"attribute": attr, "side": rng.choice(["follow", "avoid"])}
                for attr in sampled_attrs
            ]
            user = {
                "split": split,
                "user_idx": usr_idx,
                "rewrite_style_attributes": rewrite_style_attributes,
                "conversations": [],  # list of (conversation_id, original_conversation, request_indices)
            }
            for _ in range(num_conversations_per_user):
                conversation_id, conversation = conversations_with_ids[global_conv_idx % len(conversations_with_ids)]
                global_conv_idx += 1

                # Each user turn (even indices: 0, 2, 4, ...) gets a rewrite request
                request_indices = []  # (request_idx, turn_index_in_conversation)
                for i in range(0, len(conversation), 2):
                    preceding_context = conversation[:i]
                    request_idx = len(generation_requests)
                    generation_requests.append([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(
                                conversation=_format_conversation(preceding_context),
                                prompt_to_rewrite=conversation[i]["content"],
                                formatted_attributes=_format_rewrite_style(rewrite_style_attributes),
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
            rewritten_conversations.append({
                "conversation_id": conversation_id,
                "messages": rewritten,
            })
        output_users[split].append({
            "user_idx": user["user_idx"],
            "rewrite_style_attributes": user["rewrite_style_attributes"],
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