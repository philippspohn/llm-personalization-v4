from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import torch
import torch.nn.functional as F


PRINCIPLE_QUESTION_TEMPLATE = """Evaluate the response to the previous prompt in terms of whether it satisfies this principle: {principle}. Only answer Yes or No."""


class PrincipleJudge:
    model: None | AutoModelForCausalLM = None
    tokenizer: None | AutoTokenizer = None
    
    # Token IDs for Yes/No in Llama-3 tokenizer
    TOKEN_ID_YES = 9642
    TOKEN_ID_NO = 2822

    def __init__(
        self,
        model: str = "nvidia/Llama-3.3-Nemotron-70B-Reward-Principle",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        self.model_name = model
        self.torch_dtype = torch_dtype
        self.device_map = device_map

    def load_model(self) -> None:
        print(f"[PrincipleJudge] Loading model {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
        print(f"[PrincipleJudge] 'Yes' token IDs: {yes_tokens}, 'No' token IDs: {no_tokens}")
        
        if yes_tokens[0] != self.TOKEN_ID_YES:
            print(f"[PrincipleJudge] Warning: Expected Yes token {self.TOKEN_ID_YES}, got {yes_tokens[0]}")
            self.TOKEN_ID_YES = yes_tokens[0]
        if no_tokens[0] != self.TOKEN_ID_NO:
            print(f"[PrincipleJudge] Warning: Expected No token {self.TOKEN_ID_NO}, got {no_tokens[0]}")
            self.TOKEN_ID_NO = no_tokens[0]
        
        print("[PrincipleJudge] Model loaded successfully")

    def _build_messages(self, prompt: str, response: str, principle: str) -> list[dict]:
        principle_question = PRINCIPLE_QUESTION_TEMPLATE.format(principle=principle)
        
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": principle_question},
        ]

    def _compute_yes_probability(self, logits: torch.Tensor) -> float:
        yes_logit = logits[self.TOKEN_ID_YES].item()
        no_logit = logits[self.TOKEN_ID_NO].item()
        
        logits_pair = torch.tensor([yes_logit, no_logit])
        probs = F.softmax(logits_pair, dim=0)
        
        return probs[0].item()

    def judge(
        self, 
        prompts: list[str], 
        responses: list[str], 
        principles: list[str],
        batch_size: int = 4,
    ) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        all_scores: list[float] = []
        
        # Process in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            
            batch_prompts = prompts[batch_start:batch_end]
            batch_responses = responses[batch_start:batch_end]
            batch_principles = principles[batch_start:batch_end]
            
            # Build and tokenize messages for this batch
            batch_inputs = []
            for prompt, response, principle in zip(batch_prompts, batch_responses, batch_principles):
                messages = self._build_messages(prompt, response, principle)
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                batch_inputs.append(tokenized)
            
            # Pad batch inputs
            max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)
            padded_input_ids = []
            padded_attention_masks = []
            
            for inp in batch_inputs:
                seq_len = inp["input_ids"].shape[1]
                pad_len = max_len - seq_len
                
                # Left-pad for causal LM
                padded_ids = F.pad(
                    inp["input_ids"], 
                    (pad_len, 0), 
                    value=self.tokenizer.pad_token_id
                )
                padded_mask = F.pad(
                    inp["attention_mask"], 
                    (pad_len, 0), 
                    value=0
                )
                
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            # With device_map="auto", inputs go to the first device in the pipeline
            input_device = next(self.model.parameters()).device
            input_ids = torch.cat(padded_input_ids, dim=0).to(input_device)
            attention_mask = torch.cat(padded_attention_masks, dim=0).to(input_device)
            
            # Generate with output_scores to get logits
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Extract scores for each sample in batch
            # outputs.scores[0] has shape (batch_size, vocab_size)
            first_token_logits = outputs.scores[0]
            
            for i in range(first_token_logits.shape[0]):
                score = self._compute_yes_probability(first_token_logits[i])
                all_scores.append(score)
        
        return all_scores

    def unload_model(self) -> None:
        if self.model is None:
            return
        
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[PrincipleJudge] Model unloaded")
