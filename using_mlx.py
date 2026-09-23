import time

from mlx_lm import load, stream_generate

model_path = "./sarvam-1-mlx-8bit"
prompt = "how many languages can you speak?"

model, tokenizer = load(model_path)

start_time = time.perf_counter()
first_token_time = None
print("Response: ", end="", flush=True)

for response in stream_generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=100,
):
    if first_token_time is None:
        first_token_time = time.perf_counter()
        print(f"\n\nTime to first token: {first_token_time - start_time:.3f}s")
        print("Response: ", end="", flush=True)

    print(response.text, end="", flush=True)

print()
