from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer
import gc
import torch
import math
from .judge import AttributeJudge
from .prompt_templates import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE, JUDGE_USER_TEMPLATE_PROMPT_ATTRIBUTE


class WeightedRatingJudge(AttributeJudge):
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
        
        print("[WeightedRatingJudge] Judge model unloaded")


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

    def judge_manual(self, judge_prompts: list[str], return_prob_dicts: bool = False) -> list[float] | tuple[list[float], list[dict[int, float]]]:
        """
        Compute the weighted 1-10 score for a list of judge prompts. The judge prompts must already include all the instructions for the judge, including an instruction to output only the score on a scale from 1 to 10, no other text.
        """

        judge_prompts = [full_prompt + "1" for full_prompt in judge_prompts]
                    
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

        assert len(scores) == len(judge_prompts)
        if return_prob_dicts:
            return scores, prob_dicts
        return scores

    def judge_response_attribute(self, conversations: list[list[dict[str, str]]], attributes: list[str], return_prob_dicts: bool = False) -> list[float] | tuple[list[float], list[dict[int, float]]]:
        judge_prompts = []

        for messages, attribute in zip(conversations, attributes):
            message_string = ""
            if len(messages) < 2:
                raise ValueError(f"Conversation has less than 2 messages: {messages}")
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                raise ValueError(f"Last message must be an assistant response and the second to last message must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = JUDGE_SYSTEM_PROMPT
            user_prompt = JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE.format(conversation=message_string, response=messages[-1]["content"], attribute=attribute)
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            judge_prompts.append(full_prompt)
            
        return self.judge_manual(judge_prompts, return_prob_dicts)
