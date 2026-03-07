from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import torch
from typing import Any
from dataclasses import dataclass


@dataclass
class ModelResponse:
    content: str
    reasoning: str | None
    raw_text: str
    finish_reason_stop: bool

class LLMHelper:
    llm: None | LLM = None
    tokenizer: None | AutoTokenizer = None

    def __init__(self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        enable_thinking: bool = False,
        sampling_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_thinking = enable_thinking
        self.sampling_params = sampling_params or {}
        
    def load(self):
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        
        print("[LLMHelper] Model unloaded")


    def generate(self, conversations: list[list[dict[str, str]]]) -> list[ModelResponse]:
        
        inputs = []
        for messages in conversations:
            tokenized_messages = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            inputs.append(tokenized_messages)

        outputs = self.llm.generate(inputs, sampling_params=SamplingParams(**self.sampling_params))

        model_responses = []
        for output in outputs:
            raw_text = output.outputs[0].text
            if self.enable_thinking:
                think_start = raw_text.find("<think>")
                think_end = raw_text.find("</think>")
                if think_start != -1 and think_end != -1:
                    reasoning = raw_text[think_start + len("<think>"):think_end].strip()
                    content = raw_text[think_end + len("</think>"):].strip()
                else:
                    reasoning = None
                    content = raw_text
                    print(f"[LLMHelper] Warning: No <think> tag found in output: {raw_text!r}")
            else:
                reasoning = None
                content = raw_text

            model_responses.append(ModelResponse(
                content=content,
                reasoning=reasoning,
                raw_text=raw_text,
                finish_reason_stop=output.outputs[0].finish_reason == "stop",
            ))


        return model_responses