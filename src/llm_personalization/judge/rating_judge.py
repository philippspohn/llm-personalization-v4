from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer
import gc
import torch
import math


JUDGE_USER_TEMPLATE = """
Please evaluate the following interaction:

<original_prompt> {prompt} </original_prompt>
<ai_response_to_evaluate> {response} </ai_response_to_evaluate>

Your evaluation ({range_str}):"""


class RatingJudge:
    llm: None | LLM = None
    tokenizer: None | AutoTokenizer = None
    range: (int, int) = (0, 10)

    def __init__(self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        range: (int, int) = (0, 10),
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.range = range
        
    def _build_token_score_map(self, tokenizer: AutoTokenizer) -> dict[int, int]:
        token_score_map: dict[int, int] = {}
        
        for score in range(self.range[0], self.range[1] + 1):
            text = str(score)
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            token_ids = encoded["input_ids"]
            print(f"[RatingJudge] Score '{text}' -> token IDs: {token_ids}")
            
            if len(token_ids) != 1:
                raise ValueError(
                    f"Tokenizer encodes score '{text}' into {len(token_ids)} tokens (expected 1)"
                )
            
            token_score_map[int(token_ids[0])] = score
        
        return token_score_map

    def load_llm(self):
        print(f"[RatingJudge] Loading model {self.model}")
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.token_score_map = self._build_token_score_map(self.tokenizer)
        print(f"[RatingJudge] Token score map: {self.token_score_map}")


    def _build_prompt(self, prompt: str, response: str, evaluation_system_prompt: str) -> str:
        range_str = f"{self.range[0]}-{self.range[1]}"
        user_content = JUDGE_USER_TEMPLATE.format(
            range_str=range_str,
            prompt=prompt,
            response=response,
        )
        
        messages = [
            {"role": "system", "content": evaluation_system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _compute_weighted_scores(self, outputs: list[RequestOutput]) -> list[float]:
        weighted_scores: list[float] = []
        
        for output in outputs:
            first_token_logprobs = output.outputs[0].logprobs[0]
            score_probs: list[tuple[int, float]] = []
            
            for token_id, logprob_obj in first_token_logprobs.items():
                if token_id in self.token_score_map:
                    score = self.token_score_map[token_id]
                    prob = math.exp(logprob_obj.logprob)
                    score_probs.append((score, prob))
            
            if not score_probs:
                print("Warning: No score tokens found in the generated token.")
                weighted_scores.append(5.0)
                continue
            
            total_prob = sum(prob for _, prob in score_probs)
            weighted_score = sum(score * prob for score, prob in score_probs) / total_prob
            weighted_scores.append(weighted_score)
        
        return weighted_scores

    def judge(self, prompts: list[str], responses: list[str], evaluation_system_prompts: list[str]) -> list[float]:
        formatted_prompts = [
            self._build_prompt(prompt, response, evaluation_system_prompt) 
            for prompt, response, evaluation_system_prompt 
            in zip(prompts, responses, evaluation_system_prompts)
        ]
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            logprobs=20,
        )
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        weighted_scores = self._compute_weighted_scores(outputs)
        return weighted_scores
        

    def judge_manual(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        formatted_messages = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in conversations
        ]

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            logprobs=20,
        )
        outputs = self.llm.generate(formatted_messages, sampling_params)
        weighted_scores = self._compute_weighted_scores(outputs)
        return weighted_scores
        
    def unload_llm(self) -> None:
        if self.llm is None:
            return
        
        del self.llm
        del self.tokenizer
        self.llm = None
        self.tokenizer = None
        self.token_score_map = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Judge model unloaded")
