# CUDA concat crash note

This fork started from `antirez/llama.cpp-deepseek-v4-flash`, which was primarily validated on CPU and Metal. In this environment the CUDA backend failed during DeepSeek v4 Flash inference with:

```text
GGML_ASSERT(src0->type == GGML_TYPE_F32) failed
```

The failure was not caused by the GGUF file itself. The original CUDA implementation of `GGML_OP_CONCAT` in `ggml/src/ggml-cuda/concat.cu` was hardcoded for `GGML_TYPE_F32`: the kernels used `float`, the dispatcher asserted that `src0`, `src1` and `dst` were all F32, and byte offsets were converted with `/ 4`. DeepSeek v4 Flash can produce concat tensors in lower precision formats such as F16 or BF16, so CUDA reached an operation that the fork advertised as supported but only implemented for one tensor type.

CPU and Metal did not hit this exact limitation because their concat paths support the tensor types used by the model. CUDA was different: the backend support check allowed concat for non-integer tensor types, but the actual CUDA concat operator aborted as soon as the tensor type was not F32. On this machine, with CUDA enabled and layers offloaded to the GPU, that mismatch surfaced as the assertion above.

The local fix makes CUDA concat type-generic for the same-type cases needed here. The concat kernels now dispatch for F32, F16 and BF16, preserve the existing contiguous handling for dimensions 0, 1, 2, keep dimension 3 on device-to-device memcpy, and update non-contiguous concat to use the selected CUDA element type. Unsupported mixed-type or quantized concat combinations now fail with a clearer diagnostic showing `src0`, `src1` and `dst` tensor types.

This was only the first targeted compatibility fix. It removed the immediate concat crash, but slow CUDA execution still indicated that DeepSeek v4-specific graph operations were falling back to CPU or splitting the graph too often.

Following `Fringe210/llama.cpp-deepseek-v4-flash-cuda`, this tree also imports CUDA kernels and dispatcher support for the DeepSeek v4 operations:

```text
GGML_OP_DSV4_HC_SPLIT_SINKHORN
GGML_OP_DSV4_HC_WEIGHTED_SUM
GGML_OP_DSV4_HC_EXPAND
GGML_OP_DSV4_FP8_KV_QUANTIZE
GGML_OP_DSV4_ROPE_TAIL
```

Those kernels live in `ggml/src/ggml-cuda/dsv4.cu` and are wired through `ggml/src/ggml-cuda/ggml-cuda.cu`. This should reduce CPU fallback for DeepSeek v4 Flash-specific operations and improve GPU utilization compared with the original fork. It is still not a complete CUDA performance guarantee: remaining bottlenecks may come from MoE routing, quantized matmul kernels, memory bandwidth, graph scheduling, or long-context cache behavior.
