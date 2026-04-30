# Patch 1: Decode Backend Diagnostics And CPU Fallback Guard

Goal: add low-risk diagnostics that reveal where decode work runs, and optionally fail when a decode op falls back to CPU. This patch should not change model numerics.

## Why

The model loads mostly on CUDA, but runtime GPU utilization is very low. Before optimizing kernels, identify whether decode is:

- assigned to CUDA but launch-bound by many tiny kernels;
- silently split to CPU for unsupported ops;
- repeatedly copying data between CPU and CUDA;
- unable to reuse CUDA graphs because graph shape or buffers change.

## Scope

Add a diagnostic mode enabled by an environment variable or CLI flag, for example:

```sh
LLAMA_DSV4_DECODE_DIAG=1
LLAMA_DSV4_DECODE_FAIL_ON_CPU=1
```

Keep the initial patch focused on logging, not performance changes.

## Implementation Plan

1. Find the backend scheduling point in `ggml/src/ggml-backend.cpp` where graph nodes receive backend assignments.
2. Add a small summary collector for one decode graph:
   - count nodes by backend name;
   - count nodes by op name and backend;
   - count split boundaries and host-device copies if available;
   - print a short list of CPU-assigned nodes with tensor name, op, shape, and source buffers.
3. Gate the log behind `LLAMA_DSV4_DECODE_DIAG=1` so normal runs are unaffected.
4. Add a stricter mode behind `LLAMA_DSV4_DECODE_FAIL_ON_CPU=1`:
   - only inspect decode/eval graphs after model load;
   - if a non-trivial op is assigned to CPU while CUDA is available, print details and abort.
5. Avoid false positives for harmless CPU work such as tokenizer/sampler setup. The guard should focus on ggml graph compute nodes.

## Likely Files

- `ggml/src/ggml-backend.cpp`
- `src/llama-context.cpp`
- possibly `common/arg.cpp` only if using a CLI flag instead of environment variables

## Acceptance Criteria

Run:

```sh
LLAMA_DSV4_DECODE_DIAG=1 ./build/bin/llama-cli \
  -m /home/tourtools/workspace/lcpp/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  -ngl 999 -c 4096 -b 128 -ub 32 \
  --flash-attn on --fit off \
  --no-display-prompt -p "Ciao" -n 8 --temp 0
```

Expected output should include a concise decode graph summary. If CPU nodes are present, it must list enough information to locate the unsupported op or copy source.

Then run with:

```sh
LLAMA_DSV4_DECODE_FAIL_ON_CPU=1 ...
```

Expected behavior: either no abort when decode is fully CUDA, or a clear abort showing the first CPU-assigned compute op.

## Non-goals

- Do not optimize kernels in this patch.
- Do not change DeepSeek4 graph construction.
- Do not change numerical output.

## Test Post Patch

Sì, questo conferma che il fallback FLASH_ATTN_EXT era il problema grosso.

  Confronto prima/dopo:

  - FLASH_ATTN_EXT: da CUDA0=22 CPU=21 a CUDA0=22
  - split: da 130 a 88
  - copie: da 241 a 115
  - CPU compute nodes: da 65 a 44
  - generation: da 7.1 t/s a 13.6 t/s

  Rispetto al punto iniziale di circa 1.2 t/s, sei intorno a 11x di miglioramento.

  Quello che resta su CPU ora è quasi tutto:

  REPEAT CPU=43

  più:

  GET_ROWS CPU=1

  Il GET_ROWS iniziale viene da token_embd.weight in CPU_Mapped, quindi non è decode-heavy. I REPEAT i32
  sono piccoli, ma causano ancora CUDA0 -> CPU e CPU -> CUDA0, quindi sono il prossimo bersaglio sensato se
  vuoi spremere ancora qualcosa.

  La patch FA fallback invece sembra riuscita: niente più attention pesante su CPU.
