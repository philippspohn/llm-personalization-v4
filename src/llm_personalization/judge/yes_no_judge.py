from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from .judge import PrincipleJudge
import math


NEMOTRON_PRINCIPLE_QUESTION_TEMPLATE = """Evaluate the response to the previous prompt in terms of whether it satisfies this principle: {principle}. Only answer Yes or No."""


class YesNoJudge(PrincipleJudge):
    model: None | AutoModelForCausalLM = None
    tokenizer: None | AutoTokenizer = None
    
    yes_token_id: int
    no_token_id: int

    def __init__(
        self,
        model: str = "nvidia/Llama-3.3-Nemotron-70B-Reward-Principle",
        principle_question_template: str = NEMOTRON_PRINCIPLE_QUESTION_TEMPLATE,
        yes_token_id: int = 9642,
        no_token_id: int = 2822,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.principle_question_template = principle_question_template

    def load(self) -> None:
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
        
        if yes_tokens[0] != self.yes_token_id:
            print(f"[PrincipleJudge] Warning: Expected Yes token {self.yes_token_id}, got {yes_tokens[0]}")
        if no_tokens[0] != self.no_token_id:
            print(f"[PrincipleJudge] Warning: Expected No token {self.no_token_id}, got {no_tokens[0]}")
        
        print("[PrincipleJudge] Model loaded successfully")

    def unload(self) -> None:
        if self.llm is None:
            return

        del self.llm
        del self.tokenizer
        self.llm = None
        self.tokenizer = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def judge_principle(self, conversations: list[list[dict[str, str]]], principles: list[str]) -> list[float]:
        conversations_with_principle_question = [
            [*messages, 
            {"role": "user", "content": self.principle_question_template.format(principle=principle)}]
            for messages, principle in zip(conversations, principles)
        ]
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in conversations_with_principle_question
        ]
        sampling_params = SamplingParams(
            max_tokens=1,
            logprobs=20,
        )
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        scores = [self._score_output(o) for o in outputs]
        assert len(scores) == len(conversations)
        return scores

    def _score_output(self, output) -> float:
        logprobs_dict = output.outputs[0].logprobs[0]

        yes_obj = logprobs_dict.get(self.yes_token_id)
        no_obj = logprobs_dict.get(self.no_token_id)
        if yes_obj is None or no_obj is None:
            raise ValueError("Log probabilities for Yes or No are not found")

        logprob_yes, logprob_no = yes_obj.logprob, no_obj.logprob

        p_total = math.exp(logprob_yes) + math.exp(logprob_no)
        if p_total < 0.98:
            raise ValueError("Total probability is less than 0.98")

        margin = logprob_yes - logprob_no
        try:
            return 1.0 / (1.0 + math.exp(-margin))
        except OverflowError:
            return 0.0
        