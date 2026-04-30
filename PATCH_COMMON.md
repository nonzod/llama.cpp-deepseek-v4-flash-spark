# DeepSeek4 CUDA Performance Patch Context

Repository: `/home/tourtools/workspace/lcpp/llama.cpp-deepseek-v4-flash`

Model used for testing:

```sh
/home/tourtools/workspace/lcpp/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

Observed issue: output decode is very slow, around 1.2 tok/s with `llama-server` and `--parallel 1`. The priority is increasing generated token throughput. First-token latency is secondary. Avoiding hidden CPU fallback during decode is important.

Known machine/build facts:

- GPU: NVIDIA GB10, compute capability 12.1, about 124 GiB VRAM.
- Build is CUDA enabled, Release, compiled for `sm_121a`.
- CMake cache showed `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`.
- Verbose load showed `offloaded 44/44 layers to GPU`.
- Model buffers were roughly `CUDA0 model buffer size = 81687.67 MiB` and `CPU_Mapped model buffer size = 1010.00 MiB`.
- During a diagnostic `llama-bench`, GPU had about 81.8 GiB allocated but only 1-2% utilization, while CPU usage was high.

Relevant files:

- DeepSeek4 graph construction: `src/models/deepseek4.cpp`
- Custom DeepSeek4 CUDA ops: `ggml/src/ggml-cuda/dsv4.cu`
- CUDA backend op support and graph handling: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Backend scheduler and backend assignment: `ggml/src/ggml-backend.cpp`
- Context/perf/timing integration: `src/llama-context.cpp`

Current launch example:

```sh
./build/bin/llama-server \
  -m /home/tourtools/workspace/lcpp/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  -ngl 999 \
  -c 68356 \
  -b 256 \
  -ub 64 \
  --flash-attn on \
  --fit off \
  --poll 100 \
  --poll-batch 100 \
  -n 256 \
  --parallel 1 \
  --port 11434 \
  --host 0.0.0.0
```

Useful smaller repro command:

```sh
./build/bin/llama-cli \
  -m /home/tourtools/workspace/lcpp/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  -ngl 999 -c 4096 -b 128 -ub 32 \
  --flash-attn on --fit off \
  --perf \
  --no-display-prompt -p "Ciao" -n 32 --temp 0
```

Build command:

```sh
cmake --build build -j$(nproc)
```

Important constraint: this is a llama.cpp fork whose README says CUDA support is partial for DeepSeek V4 Flash. Prefer small, measurable patches. Do not rewrite large architectural pieces without profiling evidence.
