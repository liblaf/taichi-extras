## Benchmark

### Shell

- Tri Points: 20484
- Tet Points: 25006
- Frames: 90

| Method | Arch |   FPS | Time (s) |
| :----: | :--: | ----: | -------: |
|   CG   | cuda | 30.04 |     5.93 |
| Sparse | cuda | 16.69 |    10.27 |
| Sparse | x64  | 11.95 |    10.99 |

### Bunny

- Tri Points: 35828
- Tet Points: 47221
- Frames: 90

| Method | Arch |   FPS | Time (s) |
| :----: | :--: | ----: | -------: |
|   CG   | cuda | 25.33 |     7.30 |
| Sparse | x64  |  7.44 |    17.21 |
| Sparse | cuda |     - |        - |
