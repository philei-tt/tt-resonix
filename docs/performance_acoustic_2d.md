# Performance comparison of different simulations on different hardware: 

## Numpy baseline (CPU)
- baseline config: `./config/perf_baseline.json`
- I run simulation 5 times, and take biggest value:
  - `python3 ./tools/run_benchmark.py --config ./config/perf_baseline.json --executable ./fdtd/python/fdtd2d.py --n=5`


| Stencil (m) | dtype | Apple M4 Pro  | Ryzen 9 7950x |
| ----------- | ----- | ------------- | ------------- |
| 1           | fp32  | 166.68 Mpts/s | 240.16 Mpts/s |
| 2           | fp32  | 130.80 Mpts/s | 176.85 Mpts/s |
| 4           | fp32  | 95.79 Mpts/s  | 119.66 Mpts/s |
| 8           | fp32  | 64.29 Mpts/s  | 71.92 Mpts/s  |
| 1           | fp64  | 128.18 Mpts/s | 98.69 Mpts/s  |
| 2           | fp64  | 80.82 Mpts/s  | 66.24 Mpts/s  |
| 4           | fp64  | 48.84 Mpts/s  | 48.74 Mpts/s  |
| 8           | fp64  | 27.14 Mpts/s  | 31.62 Mpts/s  |

## C++ single core (CPU)

| Stencil (m) | dtype | Apple M4 Pro  | Ryzen 9 7950x |
| ----------- | ----- | ------------- | ------------- |
| 1           | fp32  | 563.76 Mpts/s | 283.53 Mpts/s |
| 2           | fp32  | 398.67 Mpts/s | 205.83 Mpts/s |
| 4           | fp32  | 237.79 Mpts/s | 160.66 Mpts/s |
| 8           | fp32  | 134.23 Mpts/s | 97.00 Mpts/s  |
| 1           | fp64  | 517.38 Mpts/s | 336.96 Mpts/s |
| 2           | fp64  | 380.49 Mpts/s | 289.38 Mpts/s |
| 4           | fp64  | 219.39 Mpts/s | 170.17 Mpts/s |
| 8           | fp64  | 122.70 Mpts/s | 106.27 Mpts/s |

WTF? Why is single precision float slower than double precision on ryzen? Why is single core perf so much worse compared to m4?

## C++ multi core (CPU)

| Stencil (m) | dtype | Apple M4 Pro   | Ryzen 9 7950x  |
| ----------- | ----- | -------------- | -------------- |
| 1           | fp32  | 1196.58 Mpts/s | 1321.41 Mpts/s |
| 2           | fp32  | 1019.66 Mpts/s | 1133.37 Mpts/s |
| 4           | fp32  | 781.91 Mpts/s  | 866.65 Mpts/s  |
| 8           | fp32  | 537.07 Mpts/s  | 697.76 Mpts/s  |
| 1           | fp64  | 1202.88 Mpts/s | 1295.42 Mpts/s |
| 2           | fp64  | 1012.01 Mpts/s | 1084.31 Mpts/s |
| 4           | fp64  | 782.20 Mpts/s  | 847.72 Mpts/s  |
| 8           | fp64  | 495.35 Mpts/s  | 673.22 Mpts/s  |

fp32 is slower sometimes than fp64 :thinkingface. I have a strong feeling of implicit cast to fp64 somewhere in the code
