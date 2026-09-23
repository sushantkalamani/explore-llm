from mlx_lm import batch_generate, load
import time
# Specify the checkpoint
model_path = "./sarvam-1-mlx-8bit"

# Load the corresponding model and tokenizer
model, tokenizer = load(model_path)

# A batch of prompts
prompts = [
    "Write a story about Einstein.",
    "Explain the theory of relativity.",
    "Describe the process of photosynthesis.",
    "What are the main causes of climate change?",
    "Write a story about Einstein.",
        "Explain the theory of relativity.",
        "Describe the process of photosynthesis.",
        "What are the main causes of climate change?"
]

# Apply the chat template and encode to tokens
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
    )
    for p in prompts
]

start_time = time.perf_counter()

# Set `verbose=True` to see generation statistics
result = batch_generate(
    model, tokenizer, prompts, verbose=True, return_prompt_caches=True, max_tokens=100
)
end_time = time.perf_counter()

total_time = end_time - start_time
total_tokens = result.stats.generation_tokens
throughput = total_tokens / total_time

print(result.texts[-1])
print(f"Total time: {total_time:.3f}s")
print(f"Total tokens generated: {total_tokens}")
print(f"Throughput: {throughput:.2f} tokens/sec")
