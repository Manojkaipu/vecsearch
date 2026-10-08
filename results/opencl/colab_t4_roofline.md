Ceilings (Tesla T4): 273 GB/s read, 7,649 GFLOP/s FP32 FMA; ridge point 28 FLOP/byte.

| backend | batch | kernel | ms | queries/s | data passes | FLOP/byte | GB/s streamed | GFLOP/s computed | bound by | % of that ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda | 1 | skinny | 6.3 | 158 | 1 | 0.5 | 242 | 121 | memory | 89% |
| cuda | 8 | skinny | 11.6 | 691 | 1 | 4.0 | 133 | 531 | memory | 49% |
| cuda | 32 | tiled | 29.5 | 1,083 | 1 | 64.0 | 52 | 3,327 | compute | 43% |
| cuda | 128 | tiled | 43.0 | 2,978 | 1 | 64.0 | 36 | 2,287 | compute | 30% |
| cuda | 512 | tiled | 179.1 | 2,858 | 4 | 64.0 | 34 | 2,195 | compute | 29% |
| cuda | 1024 | tiled | 378.8 | 2,704 | 8 | 64.0 | 32 | 2,076 | compute | 27% |
| cuda | 4096 | tiled | 1717.2 | 2,385 | 32 | 64.0 | 29 | 1,832 | compute | 24% |
| opencl | 1 | skinny | 6.3 | 158 | 1 | 0.5 | 243 | 122 | memory | 89% |
| opencl | 8 | skinny | 15.0 | 532 | 1 | 4.0 | 102 | 409 | memory | 37% |
| opencl | 32 | tiled | 31.7 | 1,008 | 1 | 64.0 | 48 | 3,098 | compute | 40% |
| opencl | 128 | tiled | 46.7 | 2,738 | 1 | 64.0 | 33 | 2,103 | compute | 27% |
| opencl | 512 | tiled | 195.6 | 2,618 | 4 | 64.0 | 31 | 2,010 | compute | 26% |
| opencl | 1024 | tiled | 410.0 | 2,498 | 8 | 64.0 | 30 | 1,918 | compute | 25% |
| opencl | 4096 | tiled | 1860.0 | 2,202 | 32 | 64.0 | 26 | 1,691 | compute | 22% |
| opencl-nosg | 1 | skinny | 6.4 | 157 | 1 | 0.5 | 242 | 121 | memory | 88% |
| opencl-nosg | 8 | skinny | 15.4 | 520 | 1 | 4.0 | 100 | 399 | memory | 37% |
| opencl-nosg | 32 | tiled | 33.5 | 956 | 1 | 64.0 | 46 | 2,938 | compute | 38% |
| opencl-nosg | 128 | tiled | 49.0 | 2,611 | 1 | 64.0 | 31 | 2,005 | compute | 26% |
| opencl-nosg | 512 | tiled | 203.2 | 2,519 | 4 | 64.0 | 30 | 1,935 | compute | 25% |
| opencl-nosg | 1024 | tiled | 426.5 | 2,401 | 8 | 64.0 | 29 | 1,844 | compute | 24% |
| opencl-nosg | 4096 | tiled | 1899.8 | 2,156 | 32 | 64.0 | 26 | 1,656 | compute | 22% |
