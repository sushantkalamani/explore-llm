# Benchmarking Sarvam-1 on Apple Silicon (M2 Pro)

observations on sarvam 1

The first run in MLX is slow because the model needs to get the weights from memory to RAM, compile the code and show result. Subsequent runs will use the weights from cache, hence we should skip considering first test to calculate TTFT

| Precision | MLX (warm avg) | Transformers |
| --------- | -------------- | ------------ |
| fp16      | ~1.5s          | ~2s          |
| 8-bit     | ~0.5s          | n/a         |
| 4-bit     | ~0.2s          | n/a          |


Tested sarvam 1 8bit batch wise

Results were interesting, from single prompt to batch of four I got ~3X more throughput but the throughput was around same or we can say hit the plateau for batch 8 where the peak memory was not that changed.
Throughput plateaus at batch 4-8, likely a compute or memory-bandwidth ceiling, not confirmed by profiling.

| Batch size | Throughput  | Peak memory |
| ---------- | ----------- | ----------- |
| 1          | 41.3 tok/s  | 2.74 GB     |
| 4          | 130.2 tok/s | 2.98 GB     |
| 8          | 128.6 tok/s | 3.14 GB     |



| Axis             | Result                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| Framework (fp16) | MLX ~1.5s TTFT vs transformers ~2s modest, ~25%                                                 |
| Quantization     | fp16 to 8-bit to 4-bit: roughly 3s to 1.5s to 0.5s TTFT, this was the dominant lever        |
| Batching         | 1 to 4: 3.1x throughput; 4 to 8: flat, likely compute or bandwidth ceiling, not profiled     |
