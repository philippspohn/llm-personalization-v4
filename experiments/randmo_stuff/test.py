from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 1. Define your model
model_id = "google/gemma-3-1b-it"

# 2. Initialize the Hugging Face tokenizer externally
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Your raw prompts (Imagine 9 million of these in your real script)
raw_prompts = [
    "Write a short poem about a server restarting.",
    "Explain Kubernetes in one sentence.",
    "What is the capital of Austria?"
]

# 4. Pre-tokenize the inputs using Hugging Face
# Note: For your 9 million prompts, this is the step you would distribute 
# across your CPU cores using Python's multiprocessing pool.
print("Pre-tokenizing prompts...")
tokenized_outputs = tokenizer(
    raw_prompts, 
    add_special_tokens=True # Ensures Gemma's expected BOS (Beginning of Sequence) tokens are added
)

# Extract the lists of integer IDs
input_ids_list = tokenized_outputs["input_ids"]

# 5. Format for vLLM batched inference
# vLLM expects a list of dictionaries, where each dict has a "prompt_token_ids" key.
vllm_inputs = [{"prompt_token_ids": ids} for ids in input_ids_list]

# 6. Initialize the vLLM Engine
print("Initializing vLLM Engine...")
llm = LLM(model=model_id)
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

# 7. Generate! 
# Because you are passing `vllm_inputs` (dicts of IDs) instead of raw strings, 
# vLLM completely skips the frontend text-rendering bottleneck.
print("Starting generation...")
outputs = llm.generate(vllm_inputs, sampling_params)

# 8. Parse your outputs
print("\n--- RESULTS ---")
for output in outputs:
    # Notice we access the original token IDs via output.prompt_token_ids
    prompt_length = len(output.prompt_token_ids)
    generated_text = output.outputs[0].text
    
    print(f"Tokens in prompt: {prompt_length}")
    print(f"Response: {generated_text.strip()}\n")