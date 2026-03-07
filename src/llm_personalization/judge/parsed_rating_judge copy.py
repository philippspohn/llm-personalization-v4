from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer
import gc
import re
import torch
from .judge import PrincipleJudge
from typing import Any


JUDGE_SYSTEM_PROMPT_TEMPLATE = """
You are an expert, impartial evaluator. Your task is to evaluate the provided text based strictly on the provided principle on a scale from 1 to 10. Output only the score, no other text."""

JUDGE_SYSTEM_PROMPT_TEMPLATE_THINKING = """
You are an expert, impartial evaluator. Your task is to evaluate the provided text based strictly on the provided principle on a scale from 1 to 10.

You must first use your internal thinking process to reason through the evaluation step-by-step. Once your reasoning is complete, your final visible output must be ONLY the numeric score (1-10), with absolutely no other words, punctuation, or text."""

JUDGE_USER_TEMPLATE_PRINCIPLE = """
Please evaluate how well the AI response adheres to the provided principle.

<conversation>
{conversation}
</conversation>
<ai_response_to_evaluate>
{response}
</ai_response_to_evaluate>
<principle>
{principle}
</principle>

To what extent does the response satisfy this principle?
Your evaluation (1-10):"""

JUDGE_USER_TEMPLATE_ATTRIBUTE = """
Please evaluate how strongly the provided attribute is exhibited in the user prompt.

<conversation>
{conversation}
</conversation>
<user_prompt_to_evaluate>
{user_prompt}
</user_prompt_to_evaluate>
<attribute>
{attribute}
</attribute>

To what extent does the user prompt exhibit this attribute?
Your evaluation (1-10):"""


class ParsedRatingJudge(PrincipleJudge):
    llm: None | LLM = None
    tokenizer: None | AutoTokenizer = None

    def __init__(self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = None,
        enable_thinking: bool = False,
        force_rating_on_thinking_timeout: bool = True,
        sampling_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.force_rating_on_thinking_timeout = force_rating_on_thinking_timeout

        self.sampling_params = sampling_params or {}

    def load(self):
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.think_end_token = self._detect_think_end_token()
        if self.enable_thinking and self.force_rating_on_thinking_timeout:
            if self.think_end_token:
                print(f"[ParsedRatingJudge] Detected think end token: {self.think_end_token!r}")
            else:
                print("[ParsedRatingJudge] Warning: force_rating_on_thinking_timeout enabled but no think end token found in vocabulary")
        
    def unload(self) -> None:
        if self.llm is None:
            return
        
        del self.llm
        del self.tokenizer
        self.llm = None
        self.tokenizer = None
        self.score_token_map = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[RatingJudge] Judge model unloaded")


    def _parse_score(self, output: RequestOutput) -> int:
        text = output.outputs[0].text.strip()
        if self.enable_thinking:
            # Use regex to find the last valid score (1-10) in the output — works regardless
            # of the model's thinking separator tag (<think>, <|thinking|>, etc.)
            matches = re.findall(r'\b(10|[1-9])\b', text)
            if matches:
                return int(matches[-1])
            print(f"Warning: No valid score (1-10) found in output: {text!r}, returning 5")
            return 5
        if not text.isdigit():
            print(f"Warning: Score is not a digit: {text!r}, returning 5")
            return 5
        return int(text)

    def _thinking_timed_out(self, output: RequestOutput) -> bool:
        """Check if generation stopped because it hit max_tokens (not because it finished naturally)."""
        return output.outputs[0].finish_reason == "length"

    def _has_think_end(self, output: RequestOutput) -> bool:
        return self.think_end_token is not None and self.think_end_token in output.outputs[0].text

    def _extract_reasoning(self, output: RequestOutput) -> str:
        """Extract the reasoning/thinking portion from model output."""
        text = output.outputs[0].text
        if self.think_end_token and self.think_end_token in text:
            return text.split(self.think_end_token, 1)[0].strip()
        return text.strip()

    def judge_manual(self, judge_prompts: list[str], return_reasoning: bool = False) -> list[int] | tuple[list[int], list[str]]:
        max_tokens = self.thinking_max_tokens if self.enable_thinking else self.max_tokens
        sampling_params = self._build_sampling_params(max_tokens)
        outputs = self.llm.generate(judge_prompts, sampling_params=sampling_params)
        for i in range(3):
            print("=" * 100)
            print(f"[ParsedRatingJudge] Output {i}: {outputs[i].outputs[0].text}")
            print("=" * 100)

        reasoning_texts = [self._extract_reasoning(o) for o in outputs] if return_reasoning else None

        can_force = (
            self.enable_thinking
            and self.force_rating_on_thinking_timeout
            and self.think_end_token is not None
        )

        if can_force:
            retry_prompts = []
            retry_indices = []
            for i, output in enumerate(outputs):
                if self._thinking_timed_out(output) and not self._has_think_end(output):
                    forced_prompt = judge_prompts[i] + output.outputs[0].text + self.think_end_token + "\n\n Rating: "
                    retry_prompts.append(forced_prompt)
                    retry_indices.append(i)

            if retry_prompts:
                print(f"[ParsedRatingJudge] {len(retry_prompts)}/{len(judge_prompts)} prompts hit thinking timeout, forcing rating...")
                retry_params = self._build_sampling_params(max_tokens=8)
                retry_outputs = self.llm.generate(retry_prompts, retry_params)
                for idx, retry_output in zip(retry_indices, retry_outputs):
                    outputs[idx] = retry_output

        scores = [self._parse_score(o) for o in outputs]
        assert len(scores) == len(judge_prompts)
        if return_reasoning:
            return scores, reasoning_texts
        return scores

    def judge_principle(self, conversations: list[list[dict[str, str]]], principles: list[str], return_reasoning: bool = False) -> list[float] | tuple[list[float], list[str]]:
        judge_prompts = []

        for messages, principle in zip(conversations, principles):
            if len(messages) < 2:
                raise ValueError(f"Conversation has less than 2 messages: {messages}")
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                raise ValueError(f"Last message must be an assistant response and the second to last message must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE
            user_prompt = JUDGE_USER_TEMPLATE_PRINCIPLE.format(conversation=message_string, response=messages[-1]["content"], principle=principle)
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            judge_prompts.append(full_prompt)
            
        return self.judge_manual(judge_prompts, return_reasoning=return_reasoning)

    def judge_prompt_attribute(self, conversations: list[list[dict[str, str]]], attributes: list[str], return_reasoning: bool = False) -> list[float] | tuple[list[float], list[str]]:
        judge_prompts = []

        for messages, attribute in zip(conversations, attributes):
            if messages[-1]["role"] != "user":
                raise ValueError(f"Last message must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE
            user_prompt = JUDGE_USER_TEMPLATE_ATTRIBUTE.format(conversation=message_string, user_prompt=messages[-1]["content"], attribute=attribute)
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            judge_prompts.append(full_prompt)
            
        return self.judge_manual(judge_prompts, return_reasoning=return_reasoning)
