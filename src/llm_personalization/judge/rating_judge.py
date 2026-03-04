from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer
import gc
import torch
import math
from .judge import PrincipleJudge


JUDGE_SYSTEM_PROMPT_TEMPLATE = """
You are an expert, impartial evaluator. Your task is to evaluate the provided text based strictly on the provided principle on a scale from 1 to 10. Output only the score, no other text."""

JUDGE_USER_TEMPLATE = """
Please evaluate the following interaction:

<conversation>
{conversation}
</conversation>
<ai_response_to_evaluate>
{response}
</ai_response_to_evaluate>
<principle>
{principle}
</principle>

How well does the response satisfy the principle?
Your evaluation (1-10):"""


class RatingJudge(PrincipleJudge):
    llm: None | LLM = None
    tokenizer: None | AutoTokenizer = None
    simple_tokenization = False

    def __init__(self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
    def _build_score_token_map(self, tokenizer: AutoTokenizer) -> dict[int, int]:
        score_token_map: dict[int, int] = {}
        
        for score in range(0, 10):
            text = str(score)
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            token_ids = encoded["input_ids"]
            
            if len(token_ids) != 1:
                raise ValueError(
                    f"Tokenization not supported: Tokenizer encodes score '{text}' into {len(token_ids)} tokens (expected 1)"
                )
            
            score_token_map[score] = int(token_ids[0])

        encoded = tokenizer("10", add_special_tokens=False,return_attention_mask=False)
        token_ids = encoded["input_ids"]
        if len(token_ids) == 1:
            self.simple_tokenization = True
            score_token_map[10] = token_ids[0]
        elif len(token_ids) == 2:
            if token_ids[0] != score_token_map[1] or token_ids[1] != score_token_map[0]:
                raise ValueError(
                    f"Tokenization not supported: Tokenizer encodes score '10' into {token_ids}"
                )
            self.simple_tokenization = False
        else:
            raise ValueError(
                f"Tokenization not supported: Tokenizer encodes score '10' into {len(token_ids)} tokens (expected 1 or 2)"
            )
        
        print(f"[RatingJudge] Tokenization: {"simple" if self.simple_tokenization else "not simple"}")
        print(f"[RatingJudge] Score token map: {score_token_map}")
        return score_token_map

    def load(self):
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
        
        self.score_token_map = self._build_score_token_map(self.tokenizer)
        print(f"[RatingJudge] Score token map: {self.score_token_map}")

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


    def _log_add_exp(self, x, y):
        """Safely adds two log probabilities."""
        if x == float('-inf'): return y
        if y == float('-inf'): return x
        
        m = max(x, y)
        return m + math.log(math.exp(x - m) + math.exp(y - m))


    def _compute_weighted_score(self, logprob_dict: dict[int, float]) -> float:
        score = 0.0
        total_prob = sum(math.exp(logprob) for logprob in logprob_dict.values())
        if total_prob < 0.9:
            print(f"Warning: Total probability is less than 0.9: {total_prob}")
        if total_prob < 0.5:
            raise ValueError(f"Total probability is less than 0.5: {total_prob}")

        for i in range(1, 11):
            score += i * math.exp(logprob_dict[i]) / total_prob
        return score


    def judge_principle(self, conversations: list[list[dict[str, str]]], principles: list[str], return_prob_dicts: bool = False) -> list[float] | tuple[list[float], list[dict[int, float]]]:
        judge_prompts = []

        for messages, principle in zip(conversations, principles):
            message_string = ""
            if len(messages) < 2:
                raise ValueError(f"Conversation has less than 2 messages: {messages}")
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                raise ValueError(f"Last message must be an assistant response and the second to last message must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE
            user_prompt = JUDGE_USER_TEMPLATE.format(conversation=message_string, response=messages[-1]["content"], principle=principle)
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_prompt += "1"
            judge_prompts.append(full_prompt)
                    
        sampling_params = SamplingParams(max_tokens=1, logprobs=20, prompt_logprobs=20)
        outputs = self.llm.generate(judge_prompts, sampling_params=sampling_params)

        scores = []
        prob_dicts = []
        for output in outputs:
            first_token_logprob = output.prompt_logprobs[-1]
            second_token_logprob = output.outputs[0].logprobs[0]
            
            if self.simple_tokenization:
                # Case A: Simple Tokenization (1 and 10 are different tokens)
                logprob_dict = {
                    nr: first_token_logprob[self.score_token_map[nr]].logprob if self.score_token_map[nr] in first_token_logprob else float('-inf')
                    for nr in range(1, 11)
                }
            else:
                # Case B: Not Simple Tokenization (token 10 is made up of two tokens ("1" and "0"))

                # 1. Initialize everything to -inf (0% probability)
                logprob_dict = {nr: float('-inf') for nr in range(1, 11)}

                # 2. Fill 2 through 9
                for nr in range(2, 10):
                    if self.score_token_map[nr] in first_token_logprob:
                        logprob_dict[nr] = first_token_logprob[self.score_token_map[nr]].logprob

                # 3. Check if we have probability for token 1 (otherwise leave at 0% probability)
                if self.score_token_map[1] in first_token_logprob:
                    # 4. Compute the probability of token 10
                    first_token_one_logprob = first_token_logprob[self.score_token_map[1]].logprob
                    if self.score_token_map[0] not in second_token_logprob:
                        logprob_dict[10] = float('-inf')
                    else:
                        logprob_dict[10] = first_token_one_logprob + second_token_logprob[self.score_token_map[0]].logprob # P("1" then "0") (joint probability)
                    
                    # 5. Compute the probability of token 1
                    # log P(any digit | "1") via log-sum-exp
                    log_prob_digit_after_one = float('-inf')
                    for d in range(0, 10):
                        digit_token_id = self.score_token_map[d]
                        if digit_token_id in second_token_logprob:
                            log_prob_digit_after_one = self._log_add_exp(
                                log_prob_digit_after_one,
                                second_token_logprob[digit_token_id].logprob,
                            )

                    # log P(score=1) = log P("1") + log(1 - P(any digit | "1"))
                    if log_prob_digit_after_one >= -1e-10:
                        # All probability mass is on digit continuations → P(score=1) ≈ 0
                        logprob_dict[1] = float('-inf')
                    else:
                        log_prob_not_digit = math.log1p(-math.exp(log_prob_digit_after_one))
                        logprob_dict[1] = first_token_one_logprob + log_prob_not_digit


                    # Limitations: some shortcuts with unlikely continuations; token 1 and 10 are less likely to be zero becuase to the top-20 logprop threshold; but practically negligible

            prob_dict = {i: math.exp(logprob_dict[i]) for i in range(1, 11)}
            prob_dicts.append(prob_dict)
            score = self._compute_weighted_score(logprob_dict)
            scores.append(score)

        assert len(scores) == len(conversations)
        if return_prob_dicts:
            return scores, prob_dicts
        return scores

    # def _build_prompt(self, prompt: str, response: str, evaluation_system_prompt: str) -> str:
    #     range_str = f"{self.range[0]}-{self.range[1]}"
    #     user_content = JUDGE_USER_TEMPLATE.format(
    #         range_str=range_str,
    #         prompt=prompt,
    #         response=response,
    #     )
        
    #     messages = [
    #         {"role": "system", "content": evaluation_system_prompt},
    #         {"role": "user", "content": user_content},
    #     ]
        
    #     return self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         enable_thinking=False,
    #     )

    

    # def _compute_weighted_scores(self, outputs: list[RequestOutput]) -> list[float]:
    #     weighted_scores: list[float] = []
        
    #     for output in outputs:
    #         first_token_logprobs = output.outputs[0].logprobs[0]
    #         score_probs: list[tuple[int, float]] = []
            
    #         for token_id, logprob_obj in first_token_logprobs.items():
    #             if token_id in self.token_score_map:
    #                 score = self.token_score_map[token_id]
    #                 prob = math.exp(logprob_obj.logprob)
    #                 score_probs.append((score, prob))
            
    #         if not score_probs:
    #             print("Warning: No score tokens found in the generated token.")
    #             weighted_scores.append(5.0)
    #             continue
            
    #         total_prob = sum(prob for _, prob in score_probs)
    #         weighted_score = sum(score * prob for score, prob in score_probs) / total_prob
    #         weighted_scores.append(weighted_score)
        
    #     return weighted_scores

    # def judge(self, prompts: list[str], responses: list[str], evaluation_system_prompts: list[str]) -> list[float]:
    #     formatted_prompts = [
    #         self._build_prompt(prompt, response, evaluation_system_prompt) 
    #         for prompt, response, evaluation_system_prompt 
    #         in zip(prompts, responses, evaluation_system_prompts)
    #     ]
    #     sampling_params = SamplingParams(
    #         temperature=1.0,
    #         max_tokens=1,
    #         logprobs=20,
    #     )
    #     outputs = self.llm.generate(formatted_prompts, sampling_params)
    #     weighted_scores = self._compute_weighted_scores(outputs)
    #     return weighted_scores
        

    # def judge_manual(self, conversations: list[list[dict[str, str]]]) -> list[float]:
    #     formatted_messages = [
    #         self.tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #             add_generation_prompt=True,
    #             enable_thinking=False,
    #         )
    #         for messages in conversations
    #     ]

    #     sampling_params = SamplingParams(
    #         temperature=1.0,
    #         max_tokens=1,
    #         logprobs=20,
    #     )
    #     outputs = self.llm.generate(formatted_messages, sampling_params)
    #     weighted_scores = self._compute_weighted_scores(outputs)
    #     return weighted_scores
        
    # def unload_llm(self) -> None:
    #     if self.llm is None:
    #         return
        
    #     del self.llm
    #     del self.tokenizer
    #     self.llm = None
    #     self.tokenizer = None
    #     self.token_score_map = None
        
    #     gc.collect()
        
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #         torch.cuda.synchronize()
        
    #     print("Judge model unloaded")
