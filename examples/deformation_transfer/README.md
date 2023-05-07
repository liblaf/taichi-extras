# Deformation Transfer

## Benchmark

### x64 `12th Gen Intel(R) Core(TM) i7-1260P (16)`

| Task               | Iterations | Time             | Kernel Time |
| ------------------ | ---------- | ---------------- | ----------- |
| `s1 -> t1`         | `131072`   | `0:01:12.125142` | `40.411 s`  |
| `u0 -> u0-aligned` | `4096`     | `0:00:03.423797` | `1.429 s`   |
| `s1 -> u1-aligned` | `131072`   | `0:01:20.373831` | `44.751 s`  |
| `all`              |            | `2m44s`          |             |

### cuda `NVIDIA GeForce RTX 3090`

| Task               | Iterations | Time             | Kernel Time |
| ------------------ | ---------- | ---------------- | ----------- |
| `s1 -> t1`         | `131072`   | `0:01:25.160106` | `9.193 s`   |
| `u0 -> u0-aligned` | `4096`     | `0:00:06.077657` | `0.453 s`   |
| `s1 -> u1-aligned` | `131072`   | `0:01:23.951527` | `8.746 s`   |
| `all`              |            | `3m18s`          |             |

### On Local Machine (i7-1260P, no dedicated GPU)

| Arch   | Time             | Compile Time | Kernel Time   |
| ------ | ---------------- | ------------ | ------------- |
| x64    | `0:01:25.128130` | `325.730 ms` | `46.535 s`    |
| vulkan | `0:20:01.814735` | `14.127 ms`  | Not Available |

### On Remote Server (E5-2683 v3, RTX 3090)

| Arch   | Time             | Compile Time | Kernel Time   |
| ------ | ---------------- | ------------ | ------------- |
| cuda   | `0:01:25.572995` | `567.996 ms` | `8.679 s`     |
| vulkan | `0:02:02.939205` | `41.185 ms`  | Not Available |
| x64    | `0:04:35.204867` | `633.295 ms` | `189.465 s`   |
