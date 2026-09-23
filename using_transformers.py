import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

QUANT_BITS = None  # None for fp16, or 8 for qint8 (4 is unsupported on this MPS/torch build)

model_name = "sarvamai/sarvam-1"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
	model_name,
	torch_dtype=torch.float16,
)
if QUANT_BITS == 8:
	from optimum.quanto import freeze, qint8, quantize

	quantize(model, weights=qint8)
	freeze(model)
model.to(device)

messages = [{"role": "user", "content": "Namaste, how are you?"}]
prompt = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
	add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = {
	**inputs,
	"streamer": streamer,
	"max_new_tokens": 100,
}
start_time = time.perf_counter()
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

first_token_time = None
print("Response: ", end="", flush=True)
for text in streamer:
	if first_token_time is None:
		first_token_time = time.perf_counter()
		print(f"\n\nTime to first token: {first_token_time - start_time:.3f}s")
		print("Response: ", end="", flush=True)
	print(text, end="", flush=True)

thread.join()
print()
print(f"Device: {device}")
print(f"Dtype: {next(model.parameters()).dtype}")