# Performance comparison of different simulations on different hardware: 

## Numpy baseline (CPU)
- baseline config: `./config/perf_baseline.json`
- I run simulation 5 times, and take biggest value:
  - `for i in {1..5}; do python3 ./fdtd/python/fdtd2d.py ./config/perf_baseline.json; done`

| Stencil (m) | Apple M4 Pro  | Ryzen 9 7950x |
| ----------- | ------------- | ------------- |
| 1           | 169.97 Mpts/s | 243.10 Mpts/s |
| 2           | 134.58 Mpts/s | 175.79 Mpts/s |
| 4           | 100.59 Mpts/s | 116.66 Mpts/s |
| 8           | 67.38 Mpts/s  | 70.19 Mpts/s  |
