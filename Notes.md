observations on sarvam 1

The first run in MLX is slow because the model needs to get the weights from memroy to RAM, complie the code and show result. Subsiquest runs will use the weignts from cache, hence we shoudl skip considering first test to calcualte TTFT

| Precision | MLX (warm avg) | Transformers |
| --------- | -------------- | ------------ |
| fp16      | ~1.5s          | ~2s          |
| 8-bit     | ~0.5s          | n/a         |
| 4-bit     | ~0.2s          | n/a          |


Tested sarvam 1 8bit batch wise

Result were intresting from single prompt to batch of four I got ~3X more thouput but the thruopt was arount same or we can say hit the plateaus for batch 8 where the peak memory was not that changed.
So we can conclude that throuput with my macbook m2pro I can get max of 4 batch size, and I am not memory bountd but gpu/compute bound

| Batch size | Throughput  | Peak memory |
| ---------- | ----------- | ----------- |
| 1          | 41.3 tok/s  | 2.74 GB     |
| 4          | 130.2 tok/s | 2.98 GB     |
| 8          | 128.6 tok/s | 3.14 GB     |



| Axis             | Result                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| Framework (fp16) | MLX ~1.5s TTFT vs transformers ~2s modest, ~25%                                                 |
| Quantization     | fp16 → 8-bit → 4-bit: roughly 3s → 1.5s → 0.5s TTFT — this was the dominant lever           |
| Batching         | 1→4: 3.1x throughput; 4→8: flat — concurrency ceiling around batch 4 on M2 Pro unified memory |
