Ceilings (Intel(R) Graphics [0x64a0]): 104 GB/s read, 3,648 GFLOP/s FP32 FMA; ridge point 35 FLOP/byte.

| backend | batch | kernel | ms | queries/s | data passes | FLOP/byte | GB/s streamed | GFLOP/s computed | bound by | % of that ceiling | distance kernel (ms) | top-k (ms) | distance kernel alone: % of ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opencl | 1 | skinny | 17.0 | 59 | 1 | 0.5 | 90 | 45 | memory | 87% | 15.4 | 0.6 | 96% |
| opencl | 8 | skinny | 18.4 | 434 | 1 | 4.0 | 83 | 333 | memory | 81% | 15.8 | 1.7 | 94% |
| opencl | 32 | tiled | 42.7 | 750 | 1 | 64.0 | 36 | 2,304 | compute | 63% | 38.9 | 3.7 | 69% |
| opencl | 128 | tiled | 64.3 | 1,991 | 1 | 64.0 | 24 | 1,529 | compute | 42% | 53.9 | 11.0 | 50% |
| opencl | 512 | tiled | 270.4 | 1,893 | 4 | 64.0 | 23 | 1,454 | compute | 40% | 225.0 | 54.6 | 48% |
| opencl | 1024 | tiled | 567.1 | 1,806 | 8 | 64.0 | 22 | 1,387 | compute | 38% | 444.0 | 134.8 | 49% |
| opencl | 4096 | tiled | 2666.7 | 1,536 | 32 | 64.0 | 18 | 1,180 | compute | 32% | 1751.2 | 983.9 | 49% |
| opencl-nosg | 1 | skinny | 16.6 | 60 | 1 | 0.5 | 92 | 46 | memory | 89% | | | |
| opencl-nosg | 8 | skinny | 23.8 | 336 | 1 | 4.0 | 65 | 258 | memory | 62% | | | |
| opencl-nosg | 32 | tiled | 42.7 | 750 | 1 | 64.0 | 36 | 2,303 | compute | 63% | | | |
| opencl-nosg | 128 | tiled | 63.1 | 2,030 | 1 | 64.0 | 24 | 1,559 | compute | 43% | | | |
| opencl-nosg | 512 | tiled | 265.4 | 1,929 | 4 | 64.0 | 23 | 1,481 | compute | 41% | | | |
| opencl-nosg | 1024 | tiled | 552.3 | 1,854 | 8 | 64.0 | 22 | 1,424 | compute | 39% | | | |
| opencl-nosg | 4096 | tiled | 2611.7 | 1,568 | 32 | 64.0 | 19 | 1,204 | compute | 33% | | | |
