# Performance comparison of different simulations on different hardware: 

## Numpy baseline (CPU)
- baseline config: `./config/perf_baseline.json`
- I run simulation 5 times, and take biggest value

| Stencil (m) | Apple M4 Pro  | Ryzen 9 7950x |
| ----------- | ------------- | ------------- |
| 1           | 169.97 Mpts/s | x             |
| 2           | 134.58 Mpts/s | x             |
| 4           | 100.59 Mpts/s | x             |
| 8           | 67.38 Mpts/s  | x             |
