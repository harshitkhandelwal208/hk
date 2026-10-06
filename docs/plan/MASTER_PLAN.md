<!--
PROVENANCE: Owner-authored long-term plan, stored verbatim on 2026-10-06 so it survives outside any
chat session. It is the TARGET, not the current state. For what exists today see STATUS.md.
Do not edit the plan body without the owner's say-so; record proposed changes in HANDOFF.md instead.
Section numbers (§N) are referenced from STATUS.md and docs/graph/curated.json.
-->

# HK — Final Implementation Master Plan

> **Status:** Canonical long-term implementation plan  
> **Scope:** Complete HK platform, runtime, tensor format, inference engine, training stack, research stack, accelerator ecosystem, SDK ecosystem, and distribution ecosystem  
> **Principle:** Nothing is considered permanently out of scope. Work may be staged, but every capability described here is an intended end-state of the project.

---

# 0. Vision

HK becomes a **universal native neural model format, tensor runtime, inference engine, training/research framework, and model lifecycle system** built around one central idea:

> **Treat model data, compute, memory, hardware, and model evolution as one system rather than separate layers.**

The final system should provide:

```text
Hugging Face / GGUF / SafeTensors / PyTorch
                    │
                    ▼
             HK conversion layer
                    │
                    ▼
                 .hk
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
    storage      execution     lineage
       │            │            │
       └────────────┼────────────┘
                    ▼
               HK Runtime
                    │
       ┌────────────┼─────────────────────┐
       │            │                     │
       ▼            ▼                     ▼
      CPU          GPU                   NPU
       │            │                     │
       │      ┌─────┼───────────┐         │
       │      │     │           │         │
       │    CUDA  ROCm/      Metal/     CoreML/
       │      │    HIP        Vulkan      QNN/
       │      │     │           │         │
       └──────┴─────┴───────────┴─────────┘
                    │
                    ▼
          inference / serving / SDKs
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       desktop              mobile
          │                   │
          ▼                   ▼
 Linux / Windows /       Android / iOS /
 macOS / BSD             iPadOS / visionOS
```

The same ecosystem must additionally support:

```text
training
fine-tuning
QLoRA
LoRA
pruning
quantization
sparsification
Net2Net growth
model lineage
self-training
self-play
code evaluation
model editing
model merging
model conversion
distributed training
distributed inference
continuous batching
speculative decoding
multimodal models
```

---

# 1. Non-Negotiable Architecture Principles

These principles govern every future subsystem.

## 1.1 One canonical model abstraction

The model layer must not know whether computation occurs on CPU, CUDA, Vulkan, Metal, ROCm, NPU, or another accelerator.

```text
Model
  ↓
Session
  ↓
Execution plan
  ↓
Backend
```

Backends own hardware-specific details.

---

## 1.2 Memory is a first-class resource

Every subsystem must understand:

```text
RAM
VRAM
unified memory
NPU memory
KV-cache memory
activation memory
scratch memory
optimizer memory
communication buffers
```

The runtime must have a memory planner capable of answering:

```text
Can this model fit?

Where should every tensor live?

When can memory be reused?

Should a tensor remain mapped?

Should a tensor be moved to another device?

Is recomputation cheaper than keeping it resident?

Should a layer be CPU or GPU resident?

Should KV cache be compressed?

Should a model be partially offloaded?
```

---

## 1.3 Zero-copy where physically possible

Preserve the core HK philosophy:

```text
mapped file
    ↓
tensor view
    ↓
kernel
```

No unnecessary:

```text
file → heap copy → temporary tensor → device copy
```

Use zero-copy or near-zero-copy mechanisms whenever the platform permits them.

---

## 1.4 Quantization remains in the execution path

Never make the architecture:

```text
quantized weights
       ↓
full dequantization
       ↓
FP32 model
       ↓
GEMM
```

unless unavoidable.

The preferred model remains:

```text
quantized weights
       ↓
fused decode / dot-product / GEMM
       ↓
result
```

---

## 1.5 Every optimization must have a benchmark

No optimization gets called "faster" without:

```text
baseline
optimized path
same hardware
same model
same workload
multiple repetitions
variance
memory measurements
correctness validation
```

---

# 2. Canonical Runtime Architecture

Final runtime structure:

```text
src/
├── format/
├── reader/
├── writer/
├── metadata/
├── appendix/
├── tokenizer/
├── chat/
├── model/
├── graph/
├── planner/
├── memory/
├── scheduler/
├── sampling/
├── kernels/
├── quant/
├── sparse/
├── cpu/
├── gpu/
│   ├── cuda/
│   ├── vulkan/
│   ├── metal/
│   ├── rocm/
│   ├── directml/
│   ├── xpu/
│   ├── opencl/
│   └── webgpu/
├── npu/
│   ├── coreml/
│   ├── nnapi/
│   ├── qnn/
│   ├── openvino/
│   └── vendor/
├── distributed/
├── server/
├── bindings/
└── platform/
```

Python becomes:

```text
python/hk/
├── modeling/
├── quantization/
├── pruning/
├── training/
├── adaptive/
├── evaluation/
├── datasets/
├── conversion/
├── research/
└── export/
```

---

# 3. `.hk` Format — Full End State

The `.hk` container must become a real universal neural-model format rather than simply another weight container.

## 3.1 Storage

Support:

### Dense

```text
F64
F32
F16
BF16
FP8 E4M3
FP8 E5M2
INT8
INT16
INT32
INT64
UINT8
UINT16
UINT32
UINT64
BOOL
```

### Quantized

```text
Q4_0
Q4_1
Q5_0
Q5_1
Q8_0
Q8_1

Q2_K
Q3_K
Q4_K
Q5_K
Q6_K
Q8_K

IQ1_S
IQ1_M
IQ2_XXS
IQ2_XS
IQ2_S
IQ3_XXS
IQ3_S
IQ4_NL
IQ4_XS

TQ1_0
TQ2_0

MXFP4
NVFP4
```

### HK-native quantization

```text
DQ4
DQ8
DQ6
DQ12
DQT
NF4
NF4 + residual
```

### Sparse storage

```text
bitmask
CSR
BSR
2:4
sparse FP16
sparse DQ8
sparse DQ4 + 2:4
```

### Virtual storage

```text
null_ref
shared_ref
lora_ref
delta_ref
external_ref
```

---

# 4. Format Evolution

Implement:

```text
version negotiation
forward-compatible metadata
backward-compatible readers
capability negotiation
optional sections
feature flags
compression
checksums
cryptographic hashes
signed manifests
authenticated model provenance
```

The current reserved checksum mechanism must become a real integrity system.

Final verification should support:

```text
file integrity
tensor integrity
metadata integrity
appendix integrity
lineage integrity
signature verification
trusted publisher verification
```

---

# 5. Crash-Safe Container Mutation

All editing operations must eventually become transactional.

Implement:

```text
atomic metadata updates
atomic appendix writes
transaction journal
copy-on-write metadata
appendix transaction markers
recovery after interrupted writes
automatic integrity repair
```

Commands:

```bash
hk verify model.hk
hk repair model.hk
hk compact model.hk
hk fsck model.hk
```

`compact` should remove:

```text
dead metadata blocks
rolled-back appendix records
unreferenced storage
unused padding
obsolete tensor versions
```

without altering logical content.

---

# 6. Storage Layout Optimization

The format must support explicit physical layout metadata.

Layouts:

```text
row-major
column-major

16x8
16x16
32x16
32x32
64x64
128x64
128x128

block sparse
2:4 sparse
backend-specific packed layouts
```

Support:

```text
CPU layout
CUDA layout
Metal layout
ROCm layout
Vulkan layout
NPU layout
```

A model may have multiple physically optimized representations while retaining one logical tensor identity.

---

# 7. Memory Mapping

Full implementation across platforms:

### Linux

```text
mmap
huge pages
NUMA-aware mapping
madvise
prefetch
readahead control
page-cache management
```

### Windows

```text
CreateFileMapping
MapViewOfFile
large pages where appropriate
prefetch
NUMA policies
```

### macOS

```text
mmap
MADV-style hints where available
16 KiB page-aware alignment
Apple unified-memory optimizations
```

### Android

```text
mmap
ashmem-compatible facilities where required
Android file APIs
asset-backed models
```

### iOS/iPadOS

```text
mmap
sandbox-aware model storage
16 KiB page alignment
memory-pressure integration
```

---

# 8. Raw Storage and Super-Coalescing

Complete the existing raw-storage direction.

Implement:

```text
page-aligned payloads
cache-line-aware tensor placement
large-page-compatible layouts
DMA-friendly allocations
device-importable buffers
backend-specific alignment
```

Where the platform permits:

```text
file-backed host memory
    ↓
device-import / external memory
```

without an intermediate copy.

---

# 9. Tensor Core and Tensor Optimization Layer

The entire previous tensor optimization work becomes part of the canonical kernel architecture.

## 9.1 GEMM

Implement:

```text
scalar fallback
SIMD GEMM
register-tiled GEMM
cache-blocked GEMM
multi-threaded GEMM
quantized GEMM
sparse GEMM
Tensor Core GEMM
NPU GEMM
```

---

## 9.2 GEMV

Keep and extend:

```text
4-row register tiling
row blocking
vectorized loads
prefetch
scale fast paths
zero elimination
unit-scale fast paths
activation reuse
```

Optimize decode specifically around the fact that:

```text
decode is generally bandwidth bound
```

---

## 9.3 Prefill

Implement dedicated batch kernels.

Target:

```text
8 tokens
16
32
64
128
256
512+
```

with specialized tile strategies.

The engine must never accidentally route a substantial prefill workload through token-by-token GEMV.

---

# 10. SIMD Architecture Matrix

## x86-64

Implement and benchmark:

```text
generic
SSE4.2
AVX2
AVX2 + FMA
AVX-VNNI
AVX512
AVX512-VNNI
AVX512-BF16 where available
AMX where available
```

Use runtime dispatch.

---

# 11. ARM

Implement:

```text
NEON
dotprod
FP16
BF16
SVE
SVE2
```

Optimize specifically for:

```text
Apple Silicon
Qualcomm
MediaTek
AWS Graviton
Ampere
Raspberry Pi
Android ARM64
```

---

# 12. RISC-V

Add:

```text
RV64
RVV
vectorized dot products
quantized kernels
```

Create a portable fallback for chips without vector extensions.

---

# 13. Additional CPU Architectures

Long-term support:

```text
LoongArch
POWER
s390x
```

through portable kernels first and architecture-specific kernels where worthwhile.

---

# 14. Tensor Math Optimizations

Integrate:

```text
FMA
vector dot product
integer dot-product instructions
FP16 arithmetic
BF16 arithmetic
FP8 arithmetic
lookup-table dequantization
vectorized scale conversion
fused activation
fused normalization
```

---

# 15. RoPE Optimization

Preserve and generalize the previous major optimization.

Implement:

```text
precomputed angle tables
per-context cache
per-head reuse
vectorized sin/cos
SIMD RoPE
GPU RoPE
fused RoPE + QK normalization
```

Supported strategies:

```text
standard RoPE
linear scaling
NTK scaling
YaRN
Llama 3 scaling
dynamic NTK
long-context interpolation
```

---

# 16. Attention Optimization

CPU:

```text
vectorized QK
vectorized softmax
online softmax
blocked attention
KV tiling
```

GPU:

```text
FlashAttention-style kernels
online softmax
shared-memory tiling
warp-specialized attention
Tensor Core attention
```

Support:

```text
MHA
MQA
GQA
sliding-window attention
local attention
global/local hybrid attention
```

Eventually:

```text
paged attention
compressed KV
KV quantization
KV eviction
KV cache reuse
```

---

# 17. Softmax Optimization

Retain:

```text
max subtraction
underflow pruning
```

and add:

```text
vectorized exp
fast approximate exp
GPU fast math
online softmax
threshold pruning
causal-mask fusion
```

All approximations must have numerical-error tests.

---

# 18. Normalization Optimization

Implement optimized versions of:

```text
RMSNorm
LayerNorm
GroupNorm
QK-Norm
```

Use:

```text
SIMD
warp reduction
shared-memory reduction
fused normalization
```

---

# 19. Activation Optimization

Support:

```text
SiLU
GELU
QuickGELU
SwiGLU
GeGLU
ReGLU
ReLU
```

Use:

```text
fused activation
vectorization
lookup tables where beneficial
fast-math paths with error bounds
```

---

# 20. FP8 Optimization

Retain the comptime LUT approach and extend it.

Implement:

```text
E4M3 LUT
E5M2 LUT
vectorized lookup
GPU LUT
Tensor Core FP8
scale metadata
per-tensor scaling
per-channel scaling
block scaling
```

---

# 21. MXFP4 / NVFP4 / Microscaling

Implement complete execution, not merely storage.

Support:

```text
packing
unpacking
scale handling
dequantization inside kernels
CPU kernels
CUDA kernels
ROCm kernels
Vulkan shaders
Metal kernels
NPU translation where supported
```

Benchmark against higher-precision baselines.

---

# 22. Quantization Research Stack

Complete:

```text
NF4
INT8
INT4
mixed precision
dual-mode quantization
residual recovery
adaptive quantization
per-channel quantization
per-group quantization
activation-aware quantization
```

Implement:

```text
calibration
error analysis
layer sensitivity
automatic format selection
mixed-format model generation
```

---

# 23. Dual-Mode Quantization

The final feature should be:

```text
base quantized tensor
        +
optional residual stream
```

with runtime policies:

```text
base only
base + residual
selective residual
adaptive residual
quality target mode
```

Example:

```bash
hk quantize model.hk \
  --mode dual \
  --target-quality 99.5
```

The runtime should choose whether the residual is worth decoding.

---

# 24. Automatic Mixed Precision / Quantization

Implement a global planner:

```text
layer sensitivity
+
hardware capabilities
+
memory budget
+
quality target
```

Then select:

```text
embedding → BF16
attention → Q8
FFN → Q4
LM head → Q6
```

or any other optimal combination.

---

# 25. Pruning Research

Complete all existing pruning directions:

```text
magnitude pruning
Wanda pruning
structured pruning
L2/channel reduction
BSR/block sparsity
2:4 pruning
mask-constrained recovery
```

Add:

```text
activation-aware pruning
gradient-aware pruning
layer-sensitive pruning
automatic sparsity schedules
structured channel pruning
head pruning
expert pruning
```

---

# 26. 2:4 Hardware Sparsity

This moves from storage-only to real execution.

Implement:

```text
2:4 packing
sparse metadata
sparse CPU kernels
CUDA sparse Tensor Core kernels
ROCm sparse paths
Vulkan sparse path where supported
```

For NVIDIA:

```text
Ampere+
sparse matrix math
structured metadata encoding
```

Benchmark:

```text
dense
dense quantized
2:4 dense precision
2:4 quantized
```

with accuracy recovery/fine-tuning.

---

# 27. Sparse Format Generalization

Implement execution support for:

```text
bitmask
CSR
BSR
N:M
block sparsity
2:4
```

The runtime should choose sparse vs dense execution automatically.

---

# 28. Dynamic Architecture Growth

Fully implement:

```text
Net2WiderNet
Net2DeeperNet
SwiGLU widening
linear-layer widening
vocabulary expansion
embedding expansion
attention-head expansion
KV-head expansion
expert expansion
layer insertion
```

Provide:

```bash
hk expand
hk diagnose-growth
hk plan-growth
hk apply-growth
```

---

# 29. Function Preservation

Every growth operation must have a formal verification test.

Check:

```text
pre-growth logits
post-growth logits
maximum absolute difference
cosine similarity
token agreement
```

before training.

---

# 30. Dynamic Growth During Training

Complete the existing adaptive-growth mechanism:

```text
plateau detector
evaluation-based trigger
capacity governor
hardware budget
parameter budget
memory budget
growth ratio
growth cooldown
```

Support growth of:

```text
width
depth
vocabulary
attention capacity
experts
context capacity
```

---

# 31. Plasticity Isolation

Complete:

```text
base-weight freezing
gradient masks
old-neuron protection
new-capacity-only training
selective unfreezing
scheduled plasticity release
```

Add evaluation for:

```text
old knowledge retention
new-task improvement
catastrophic forgetting
capacity utilization
```

---

# 32. Self-Conversation Research

Turn the current self-conversation component into a complete framework.

Roles:

```text
proposer
thinker
coder
critic
verifier
reflector
judge
```

Pipeline:

```text
problem
 ↓
proposal
 ↓
reasoning
 ↓
implementation
 ↓
execution
 ↓
test
 ↓
critique
 ↓
revision
 ↓
verification
```

Successful traces become training examples only after validation.

---

# 33. Code Evaluation / Sandbox

Build a real security architecture.

Support:

```text
local subprocess
container
VM
microVM
seccomp
namespaces
network isolation
filesystem isolation
CPU limit
memory limit
wall-clock timeout
process limit
disk quota
syscall policy
```

Never silently fall back from secure execution to unsafe execution.

---

# 34. Self-Training

Complete the full pipeline:

```text
generate
 ↓
test
 ↓
verify
 ↓
filter
 ↓
score
 ↓
train
 ↓
evaluate
 ↓
rollback on regression
 ↓
record lineage
```

Support:

```text
curriculum learning
automatic curriculum generation
domain-specific curricula
difficulty estimation
failure mining
retraining on failures
```

---

# 35. Self-Play / SPIN

Implement full self-play rather than merely exposing helper objects.

Support:

```text
generation A
generation B
preference construction
SPIN-style loss
LoRA evolution
validation
regression detection
rollback
lineage recording
```

Add:

```text
reference model snapshots
KL regularization
reward thresholds
anti-collapse evaluation
```

---

# 36. Autonomous Expansion

Complete the loop:

```text
model evaluation
      ↓
capacity diagnosis
      ↓
hardware budget
      ↓
growth proposal
      ↓
function-preserving expansion
      ↓
training
      ↓
evaluation
      ↓
accept / reject
      ↓
lineage record
```

No expansion should become permanent without evaluation.

---

# 37. Model Lineage

The appendix becomes a full model version-control system.

Support:

```text
hash chain
signatures
model identities
parent/child relationships
branching
merging
rollback
delta patches
LoRA records
topology records
evaluation records
code-evaluation records
training records
dataset provenance
```

Commands:

```bash
hk history model.hk
hk diff model.hk
hk branch model.hk
hk merge model.hk
hk rollback model.hk
hk verify-lineage model.hk
hk sign model.hk
hk verify-signature model.hk
```

---

# 38. Model Version Graph

Move beyond a simple linear chain.

Final model history should resemble:

```text
base
 ├── math-v1
 │    ├── math-v2
 │    └── code-v1
 │
 └── code-v1
      └── merged-v1
```

The runtime must know which materialized model state is being executed.

---

# 39. Model Deltas

Implement:

```text
tensor deltas
LoRA deltas
layer additions
sparse deltas
quantization deltas
metadata deltas
```

Allow:

```text
base + adapter
base + adapter + adapter
base + delta
merged materialization
```

---

# 40. LoRA / QLoRA

Fully implement:

```text
LoRA
QLoRA
rank scheduling
adapter stacking
adapter merging
adapter composition
adapter quantization
adapter offloading
adapter selection at runtime
```

Support inference without merging adapters into the base model.

---

# 41. Training Framework

Evolve `HKTrainer` into a real framework.

Support:

```text
pretraining
SFT
instruction tuning
continued pretraining
full fine-tuning
LoRA
QLoRA
preference tuning
DPO-style training
SPIN
self-training
distillation
pruning-aware training
quantization-aware training
```

---

# 42. Training Performance

Support:

```text
mixed precision
gradient checkpointing
activation checkpointing
gradient accumulation
distributed data parallel
tensor parallel
pipeline parallel
FSDP-like sharding
optimizer state sharding
CPU offload
NVMe offload
```

---

# 43. Training Optimizers

Support:

```text
SGD
Adam
AdamW
Adafactor
8-bit optimizers
paged optimizers
memory-efficient optimizers
```

Add scheduler support:

```text
constant
linear
cosine
warmup
one-cycle
custom callbacks
```

---

# 44. Distillation

Add:

```text
logit distillation
hidden-state distillation
attention distillation
feature distillation
progressive distillation
quantization-aware distillation
teacher-student ensembles
```

---

# 45. Dataset System

Build native dataset utilities for:

```text
streaming datasets
local datasets
Hugging Face datasets
JSONL
Parquet
Arrow
text
instruction datasets
code datasets
multimodal datasets
```

Add:

```text
shuffling
packing
dynamic batching
bucketing
token-count balancing
curriculum scheduling
```

---

# 46. Model Architecture Support

The architecture registry becomes one of the project's major systems.

Do not hard-code architecture logic throughout the runtime.

Build:

```text
ArchitectureDescriptor
ArchitectureParser
ArchitectureValidator
ArchitectureExecutor
```

Each architecture declares:

```text
embedding
attention
normalization
RoPE
KV layout
MLP
activation
MoE
output head
tokenizer requirements
```

---

# 47. Architecture Coverage Target

Reach the project's intended **137+ architecture coverage** through reusable architecture families rather than 137 unrelated implementations.

Target families include:

```text
Llama
Mistral
Mixtral
Qwen
Gemma
Gemma 2
Gemma 3
Phi
Phi-3
Phi-4
Falcon
GPT-style models
OPT-style models
BLOOM-style models
GPT-NeoX
GPT-J
MPT
StarCoder
CodeLlama
DeepSeek
Yi
InternLM
Baichuan
GLM
Command-style models
OLMo
Solar
SmolLM
TinyLlama
RWKV
Mamba
Mamba2
Jamba
Hyena-style models
MoE families
vision-language families
audio-language families
speech models
embedding models
sequence-classification models
```

The final compatibility suite must be data-driven.

---

# 48. Modern Attention Variants

Support:

```text
MHA
MQA
GQA
sliding-window attention
global/local attention
grouped rotary embeddings
multi-head latent attention
compressed KV
linear attention
state-space layers
hybrid attention
```

---

# 49. MoE

Implement:

```text
expert routing
top-k routing
capacity factor
expert parallelism
load balancing
expert caching
expert prefetch
expert quantization
expert sparsity
```

Optimize routing separately for:

```text
CPU
CUDA
ROCm
Metal
Vulkan
NPU
```

---

# 50. Long Context

Support:

```text
4K
8K
16K
32K
64K
128K
256K
1M+
```

where model architecture permits it.

Implement:

```text
KV paging
KV compression
KV quantization
attention sparsity
sliding window
chunked prefill
memory-aware context management
```

---

# 51. Speculative Decoding

Implement:

```text
draft model
target model
verification batch
accept/reject loop
dynamic draft length
```

Support:

```text
same-device speculative decoding
CPU draft + GPU target
NPU draft + GPU target
small model + large model
```

---

# 52. Sampling

Complete sampler architecture:

```text
greedy
temperature
top-k
top-p
min-p
typical sampling
repetition penalty
frequency penalty
presence penalty
mirostat
custom logits processors
```

Move supported sampling operations to GPU/NPU for low latency.

---

# 53. Grammar / Structured Output

Implement:

```text
JSON grammar
JSON Schema
regex
CFG
tool-call grammar
typed output
```

Support:

```text
streaming structured output
partial validation
incremental parser state
```

---

# 54. Multimodal Support

Expand beyond text.

## Vision

Support:

```text
image encoder
vision transformer
patch embeddings
multimodal projector
image tokens
vision-language fusion
```

## Audio

Support:

```text
speech encoder
audio embeddings
speech-to-text
audio-language models
text-to-speech interfaces
```

## Video

Support:

```text
frame sampling
temporal encoder
video tokens
vision-language generation
```

---

# 55. Multimodal Format

Extend `.hk` metadata and tensor schemas for:

```text
vision towers
audio encoders
projectors
processors
feature extractors
multimodal tokenizers
preprocessing graphs
```

---

# 56. Backend Matrix — CPU

Final CPU targets:

```text
x86-64
ARM64
RISC-V64
POWER
LoongArch
```

with portable fallback kernels.

---

# 57. Backend Matrix — NVIDIA CUDA

Complete:

```text
CUDA
PTX
CUDA Graphs
Tensor Cores
FP16
BF16
FP8
INT8
INT4
2:4 sparse Tensor Cores
```

Support:

```text
Ampere
Ada
Hopper
Blackwell
future architectures through capability-based dispatch
```

---

# 58. Backend Matrix — AMD

Implement:

```text
ROCm
HIP
MFMA
WMMA where available
FP16
BF16
FP8
INT8
quantized kernels
```

Support:

```text
RDNA
CDNA
future AMD architectures
```

---

# 59. Backend Matrix — Apple

Implement:

```text
Metal
Metal Performance Shaders
Metal compute
Apple GPU family specialization
```

Support:

```text
Apple Silicon
macOS
iOS
iPadOS
visionOS
```

Use unified-memory-aware execution.

Where available, investigate:

```text
Core ML
MPSGraph
Apple Neural Engine
```

while keeping native Metal as the performance reference path.

---

# 60. Backend Matrix — Vulkan

Maintain Vulkan as the primary portable GPU backend.

Support:

```text
NVIDIA
AMD
Intel
Apple/MoltenVK
Android GPUs
Linux GPUs
Windows GPUs
```

Use:

```text
subgroups
cooperative matrices
descriptor indexing
pipeline caching
specialization constants
device-local memory
async queues
timeline semaphores
```

---

# 61. Backend Matrix — Windows

Support:

```text
DirectML
D3D12 compute
Windows ML
```

Use DirectML/Windows ML as compatibility paths and native D3D12 where practical.

Targets:

```text
NVIDIA
AMD
Intel
Qualcomm Windows ARM
Windows NPUs
```

---

# 62. Backend Matrix — Intel

Implement:

```text
oneAPI
SYCL
Intel XPU
OpenVINO
Intel NPU
```

Support:

```text
integrated GPU
discrete Arc
CPU acceleration
NPU
```

---

# 63. Backend Matrix — Qualcomm

Implement Android/mobile acceleration through:

```text
QNN
Qualcomm AI Engine
Hexagon NPU
Vulkan
OpenCL where required
```

Support CPU fallback.

---

# 64. Backend Matrix — Android

First-class Android support:

```text
arm64-v8a
x86_64
```

Provide:

```text
AAR
JNI
Kotlin API
Java API
NDK C API
```

Backend priority:

```text
Qualcomm QNN
Android NNAPI
Vulkan
CPU NEON
```

Model storage should support:

```text
app-private storage
external model directory
Android assets
streaming download
```

---

# 65. iOS / iPadOS

Provide:

```text
Swift API
Objective-C API
C API
XCFramework
Swift Package Manager
CocoaPods if needed
```

Backends:

```text
Metal
MPS/MPSGraph
Core ML
Apple Neural Engine where appropriate
CPU NEON
```

Handle:

```text
memory pressure
background suspension
thermal throttling
battery-aware execution
16 KiB pages
sandbox restrictions
```

---

# 66. macOS

Support:

```text
Apple Silicon
Intel Macs
```

Backends:

```text
Metal
CPU
Vulkan through MoltenVK
```

Optimize for:

```text
unified memory
P-core/E-core scheduling
memory pressure
Metal zero-copy possibilities
16 KiB page layout
```

---

# 67. Linux

Support:

```text
x86_64
aarch64
RISC-V where practical
```

Backends:

```text
CUDA
ROCm
Vulkan
OpenCL
oneAPI
CPU
```

Integrate:

```text
NUMA
huge pages
cgroups
container deployment
systemd services
```

---

# 68. Windows

Support:

```text
x86_64
ARM64
```

and:

```text
CUDA
Vulkan
DirectML
D3D12
CPU
```

Provide:

```text
MSIX
installer
portable ZIP
DLL
NuGet packages
```

---

# 69. BSD / Unix

Provide best-effort:

```text
FreeBSD
OpenBSD
NetBSD
```

through:

```text
portable CPU
Vulkan where supported
```

---

# 70. WebAssembly / Browser

Build a WASM runtime:

```text
WASM SIMD
WebAssembly threads
WebGPU
IndexedDB/file-backed model storage
streaming model downloads
```

Offer:

```text
npm package
browser API
Node.js native/WASM fallback
```

---

# 71. Embedded / Edge

Long-term targets:

```text
Raspberry Pi
Jetson
edge ARM devices
industrial ARM
microserver ARM
```

Optimize:

```text
memory footprint
binary size
cold-start time
power
thermals
```

---

# 72. Backend Auto-Selection

Final selection algorithm:

```text
available device
        ↓
capabilities
        ↓
model requirements
        ↓
memory requirement
        ↓
kernel availability
        ↓
benchmark/tuning profile
        ↓
best backend
```

Example:

```text
RTX 3050
→ CUDA

RX 7800
→ ROCm

Intel Arc
→ Vulkan/XPU

Apple M-series
→ Metal

Android Snapdragon
→ QNN/Vulkan

unsupported accelerator
→ CPU
```

---

# 73. Heterogeneous Execution

Support mixed execution:

```text
CPU + CUDA
CPU + Vulkan
CPU + Metal
CPU + NPU
GPU + NPU
multiple GPUs
multiple NPUs
```

The planner should partition work based on:

```text
memory
bandwidth
latency
compute capability
transfer cost
```

---

# 74. Multi-GPU

Implement:

```text
tensor parallelism
pipeline parallelism
expert parallelism
data parallel inference
KV sharding
weight sharding
```

Support:

```text
NVLink
PCIe
xGMI
Apple unified memory
Ethernet
InfiniBand
```

where applicable.

---

# 75. Distributed Inference

Implement:

```text
worker discovery
RPC
model sharding
distributed KV
synchronization
failure recovery
load balancing
```

Support:

```text
single host
multi-host
LAN
cloud
cluster
```

---

# 76. Network Protocol

Provide a native HK inference protocol in addition to OpenAI compatibility.

Support:

```text
streaming tokens
binary tensors
remote KV transfer
model metadata
capability negotiation
authentication
encryption
compression
```

---

# 77. Server

Final `hk serve`:

```text
OpenAI compatible
Anthropic-compatible adapter
native HK protocol
SSE
WebSocket
HTTP/2
HTTP/3 where practical
```

Features:

```text
authentication
rate limits
request cancellation
streaming
metrics
logging
tracing
health checks
load balancing
```

---

# 78. Continuous Batching

Implement production scheduling:

```text
request arrival
      ↓
prefill queue
      ↓
batched prefill
      ↓
decode scheduler
      ↓
continuous batch
      ↓
completion
```

Support:

```text
priority
deadlines
tenant quotas
fair scheduling
GPU-aware scheduling
```

---

# 79. Prefix / Prompt Cache

Implement:

```text
content-addressed prefix cache
persistent prefix cache
memory-bounded cache
LRU/LFU policies
multi-tenant isolation
```

---

# 80. KV Cache

Support:

```text
segmented KV
paged KV
KV quantization
KV compression
prefix sharing
cross-request sharing
KV offload
KV eviction
```

---

# 81. Power / Thermal Awareness

For mobile and laptops:

```text
temperature
power budget
battery level
thermal state
frequency throttling
```

Allow policies:

```text
maximum performance
balanced
battery
quiet
thermal-safe
```

---

# 82. Compiler / Kernel Toolchain

Build a first-class kernel pipeline:

```text
kernel source
      ↓
architecture compiler
      ↓
binary / PTX / SPIR-V / Metal library
      ↓
embedded or cached artifact
```

Support:

```text
offline builds
runtime compilation where legal
precompiled kernels
kernel caches
specialization
autotuning
```

---

# 83. Auto-Tuning

Every major backend should have tunable parameters:

```text
tile size
warp count
vector width
shared memory usage
pipeline depth
batch threshold
unroll factor
workgroup size
```

Generate machine profiles.

---

# 84. Profiling

Provide one unified profiler:

```bash
hk profile model.hk
```

Output:

```text
operator
CPU time
GPU time
memory traffic
cache behavior
kernel launches
VRAM
RAM
power
```

Backend-specific integrations:

```text
perf
Nsight
Metal Instruments
Xcode GPU tools
ROCm profiler
Intel VTune
Android profiling
OpenVINO profiling
```

---

# 85. Benchmark Suite

Benchmark all of:

```text
startup
file mapping
first touch
warm load
prefill
decode
long context
batching
concurrency
memory
VRAM
KV growth
power
```

Compare against:

```text
llama.cpp
MLX
ONNX Runtime
TensorRT-LLM where relevant
vLLM where relevant
Transformers
Ollama
other architecture-specific reference implementations
```

Never compare incompatible configurations.

---

# 86. Performance Objectives

The goal is not one universal "faster than X" number.

Target:

```text
CPU:
within competitive range of highly optimized native runtimes

NVIDIA:
competitive with optimized CUDA inference runtimes

AMD:
competitive with ROCm-native runtimes

Apple:
competitive with Metal/MLX-native runtimes

mobile:
competitive with vendor-native execution paths

memory:
remain materially competitive through mmap + planner + quantization
```

---

# 87. Python Performance

Retain and expand the prior optimizations:

```text
cached transposed weights
version-aware weight caches
native dispatch
zero unnecessary NumPy allocations
zero-copy tensor views
batched native calls
native quantization
native GEMM
```

The Python layer should eventually be a high-level control interface, not the hot path.

---

# 88. Native Python Execution

Implement:

```text
NativeHKEngine
NativeHKTokenizer
NativeHKTensor
NativeHKModel
NativeHKTrainer
```

with:

```text
buffer protocol
DLPack
NumPy interoperability
PyTorch interoperability
CUDA tensor interoperability
```

---

# 89. PyTorch Integration

Support:

```python
model = hk.load(...)
```

and:

```python
hk_model(...)
```

as naturally as possible.

Implement:

```text
torch.Tensor ↔ HK tensor
DLPack
device transfer
autograd where applicable
training hooks
```

---

# 90. Python Export / Import

Support:

```text
PyTorch
SafeTensors
GGUF
HK
ONNX where applicable
```

round-trip conversion tests.

---

# 91. Multi-Language SDKs

Complete:

```text
C
C++
Rust
Go
C#
Java
Kotlin
TypeScript
JavaScript
Python
Swift
Objective-C
```

Every binding gets:

```text
reader
writer
model
tokenizer
inference
streaming
sampling
server client
errors
lifecycle
```

---

# 92. Package Ecosystem

Publish:

```text
PyPI
crates.io
npm
NuGet
Maven Central
Gradle/AAR
Swift Package Manager
vcpkg
Conan
Homebrew
winget
Linux packages
```

---

# 93. Model Hub Integration

`hk pull` becomes a complete model-management tool.

Support:

```text
Hugging Face
self-hosted HK registry
S3-compatible storage
HTTP
local registries
```

Features:

```text
resume
parallel download
checksums
signatures
shard handling
range requests
conversion while streaming
local cache
deduplication
```

---

# 94. Model Registry

Create:

```text
model identity
versions
architectures
quantizations
hardware targets
checksums
signatures
lineage
benchmark results
```

---

# 95. Model Conversion

Full bidirectional conversion:

```text
SafeTensors → HK
GGUF → HK
PyTorch → HK
HK → SafeTensors
HK → GGUF
HK → PyTorch
```

Preserve:

```text
metadata
tokenizer
chat templates
quantization
architecture
provenance
```

where target format supports it.

---

# 96. Sharding

Support:

```text
single-file
multi-file
streaming shards
parallel shard loading
lazy shard loading
distributed shard loading
```

---

# 97. Model Editing GUI

Turn the existing GUI into a proper model editor.

Features:

```text
model explorer
tensor browser
metadata editor
quantization controls
sparsity editor
growth planner
lineage viewer
benchmark runner
diff viewer
model merger
adapter manager
```

---

# 98. Tensor Inspection

CLI:

```bash
hk inspect
hk tensor list
hk tensor info
hk tensor stats
hk tensor diff
hk tensor export
```

Statistics:

```text
shape
dtype
quantization
sparsity
min/max
mean
std
norm
entropy
histogram
```

---

# 99. Model Validation

Implement:

```text
shape validation
dtype validation
architecture validation
tokenizer validation
metadata validation
checksum validation
lineage validation
numerical parity testing
```

---

# 100. Security

The final system must treat models as untrusted inputs.

Implement:

```text
strict bounds checks
integer overflow checks
safe decompression
resource limits
sandboxing
signed packages
trusted registries
signature verification
secure model loading
malicious-file tests
fuzzing
```

For server:

```text
authentication
authorization
tenant isolation
request limits
upload limits
DoS protection
```

---

# 101. Fuzzing

Fuzz:

```text
format parser
metadata parser
tensor TOC
quantization decoders
tokenizer
chat templates
HTTP parser
model downloader
appendix records
lineage
```

Use:

```text
AFL++
libFuzzer
cargo-fuzz where appropriate
OSS-Fuzz integration
```

---

# 102. Correctness Infrastructure

Every backend must share the same reference tests.

Reference:

```text
Python / high precision
```

Compare:

```text
CPU
CUDA
ROCm
Metal
Vulkan
NPU
```

Metrics:

```text
absolute error
relative error
cosine similarity
top-k agreement
perplexity
generation agreement
```

---

# 103. Numerical Stability

Test:

```text
FP32
FP16
BF16
FP8
INT8
INT4
quantized attention
sparse attention
long context
large logits
small logits
```

---

# 104. Model Quality Validation

Create a standard evaluation suite:

```text
perplexity
MMLU-style evaluations
coding evaluations
math evaluations
reasoning evaluations
long-context tests
tokenizer parity
structured-output tests
tool-use tests
```

For multimodal models:

```text
vision benchmarks
audio benchmarks
OCR
captioning
VQA
speech recognition
```

---

# 105. Training Research Evaluation

For every research technique:

```text
baseline
variant
training cost
memory
speed
quality
stability
forgetting
reproducibility
```

No research feature is considered complete just because it has code.

It must have:

```text
mathematical definition
implementation
unit tests
integration tests
ablation
benchmark
failure analysis
documentation
```

---

# 106. Research: Adaptive Growth

Measure:

```text
function preservation
training convergence
capacity utilization
parameter efficiency
catastrophic forgetting
hardware fit
```

---

# 107. Research: 2:4 Sparsity

Measure:

```text
compression ratio
accuracy loss
fine-tuning recovery
dense vs sparse throughput
power
memory bandwidth
Tensor Core utilization
```

---

# 108. Research: Dual-Mode Quantization

Measure:

```text
base-only quality
base+residual quality
storage size
decode cost
memory bandwidth
quality/byte
quality/token/sec
```

---

# 109. Research: Self-Training

Measure:

```text
pass rate
training loss
generalization
self-reinforcement errors
diversity collapse
evaluation leakage
regression
lineage
```

---

# 110. Research: Self-Play / SPIN

Measure:

```text
preference quality
training stability
generation quality
mode collapse
regression rate
adapter size
compute cost
```

---

# 111. Research: Automated Expansion

Measure:

```text
when expansion triggers
whether expansion was actually needed
new capacity utilization
quality gain per parameter
memory cost
training cost
forgetting
```

---

# 112. Research: Model Evolution

Build a complete model-evolution loop:

```text
evaluate
 ↓
diagnose
 ↓
propose
 ↓
edit
 ↓
train
 ↓
benchmark
 ↓
accept/reject
 ↓
commit lineage
```

Eventually support automatic experimentation across:

```text
quantization
sparsity
width
depth
vocabulary
LoRA
optimizer
learning rate
context
architecture variants
```

---

# 113. Research: Architecture Search

Add constrained search over:

```text
hidden size
layer count
FFN size
attention heads
KV heads
RoPE parameters
activation
normalization
expert count
expert width
sparsity
quantization
```

Search objective:

```text
quality
+
latency
+
memory
+
power
+
model size
```

---

# 114. Research: Hardware-Aware Model Design

The final framework should be capable of answering:

```text
What architecture gives the best quality
under 6 GB VRAM?

What quantization gives the best score
under 4 GB RAM?

What width can this device train?

What layer count fits a mobile thermal budget?
```

This combines:

```text
architecture growth
quantization
sparsity
benchmarking
hardware profiling
```

---

# 115. Distributed Training

Implement:

```text
data parallel
tensor parallel
pipeline parallel
expert parallel
FSDP-style sharding
ZeRO-style optimizer partitioning
```

Support:

```text
single machine
multi-GPU
multi-node
cloud clusters
```

---

# 116. Distributed Model Evolution

Model lineage must work across workers.

Support:

```text
distributed experiment IDs
worker provenance
dataset provenance
checkpoint lineage
merge conflict detection
reproducible experiment manifests
```

---

# 117. Experiment Tracking

Native experiment metadata:

```text
experiment ID
git commit
HK version
model hash
dataset hash
training config
hardware
backend
metrics
random seeds
lineage
```

Store it in the `.hk` appendix and optionally export to JSON/JSONL.

---

# 118. Reproducibility

Every training and optimization experiment should record:

```text
random seeds
software versions
compiler versions
kernel version
driver version
hardware
model hash
dataset hash
configuration
```

---

# 119. Mobile Runtime Architecture

Provide separate mobile execution policy.

Mobile runtime must handle:

```text
foreground/background
thermal pressure
battery
memory pressure
model eviction
cold start
incremental model loading
```

Potential policies:

```text
max speed
balanced
battery saver
thermal saver
```

---

# 120. Mobile Model Packaging

Provide:

```text
.hk
.hkpack
compressed model bundles
downloadable shards
asset bundles
```

Support:

```text
encrypted/private models
license metadata
signature verification
offline execution
```

---

# 121. Android UI Integration

Ship:

```text
Kotlin API
Jetpack-friendly wrapper
streaming callbacks
coroutines support
```

Example architecture:

```text
Android app
   ↓
HK Kotlin API
   ↓
JNI
   ↓
native HK
   ↓
QNN / Vulkan / CPU
```

---

# 122. Apple SDK Integration

Ship:

```text
Swift async generation
streaming
Metal backend
Core ML bridge
memory-pressure callbacks
```

Support:

```text
macOS
iOS
iPadOS
visionOS
```

---

# 123. Desktop Applications

Optional official GUI clients:

```text
Windows
Linux
macOS
```

Use the same native runtime instead of reimplementing inference.

---

# 124. CLI

Final CLI should include:

```bash
hk pull
hk run
hk chat
hk serve

hk inspect
hk verify
hk repair
hk compact

hk convert
hk export
hk import

hk quantize
hk prune
hk sparse

hk expand
hk diagnose-growth

hk appendix
hk history
hk diff
hk merge
hk rollback
hk sign

hk benchmark
hk profile

hk model
hk tokenizer

hk distributed
hk worker
hk cluster
```

---

# 125. Server API

Expose:

```text
/v1/models
/v1/completions
/v1/chat/completions
/v1/embeddings
/v1/rerank
/v1/audio/*
/v1/images/*
/v1/responses
```

where the backend supports the capability.

---

# 126. Embeddings

Support embedding models separately from causal LLMs.

Optimize:

```text
batch embeddings
pooling
mean pooling
CLS pooling
normalized output
GPU batching
```

---

# 127. Reranking

Support:

```text
cross encoders
pairwise ranking
batch reranking
```

---

# 128. Speech

Add:

```text
speech-to-text
audio encoder
text-to-speech adapters
streaming audio
```

using appropriate native backends.

---

# 129. Computer Vision

Support:

```text
image classification
embedding models
vision-language models
OCR
image generation interfaces
```

---

# 130. Model Pipelines

Create composable native pipelines:

```text
Tokenizer
 → Encoder
 → Retriever
 → Reranker
 → Generator
```

or:

```text
Audio
 → Speech recognition
 → LLM
 → TTS
```

---

# 131. Pipeline Graph Runtime

Introduce an internal graph:

```text
input
 ↓
operator
 ↓
operator
 ↓
operator
 ↓
output
```

This graph should be capable of:

```text
fusion
constant folding
memory planning
device placement
kernel selection
parallel scheduling
```

---

# 132. Graph Compilation

Build a backend-independent IR:

```text
HK IR
```

with:

```text
tensor shapes
dtype
quantization
device
layout
operator
memory lifetime
```

Then compile:

```text
HK IR
 ↓
CPU
CUDA
ROCm
Metal
Vulkan
NPU
```

---

# 133. Kernel Fusion at IR Level

Automatically fuse:

```text
RMSNorm → GEMM
GEMM → activation
QK-Norm → RoPE
Gate → Up → SwiGLU
attention → projection
residual → norm
```

where hardware supports it.

---

# 134. Memory Scheduling

Build liveness analysis:

```text
tensor created
tensor used
tensor dead
```

Then reuse buffers aggressively.

---

# 135. Recomputation

For memory-constrained training/inference graphs:

```text
keep
vs
recompute
```

must be decided dynamically.

---

# 136. Operator Capability System

Every operator must declare:

```text
supported dtype
supported quantization
supported backend
supported shape
supported architecture
```

This prevents runtime surprises.

---

# 137. Automatic Fallback

If a GPU lacks a kernel:

```text
optimized GPU
 ↓
portable GPU
 ↓
CPU optimized
 ↓
CPU generic
```

with explicit diagnostics.

---

# 138. Platform Testing Matrix

Final CI/device testing:

## Linux

```text
x86_64
aarch64
RISC-V where practical
CUDA
ROCm
Vulkan
Intel
```

## Windows

```text
x86_64
ARM64
CUDA
DirectML
Vulkan
```

## macOS

```text
Apple Silicon
Intel
Metal
Vulkan/MoltenVK
```

## Android

```text
ARM64
x86_64
Qualcomm
MediaTek
Samsung
Mali
Adreno
Vulkan
QNN
NNAPI
```

## iOS/iPadOS

```text
arm64
Metal
Core ML
ANE where possible
```

## Web

```text
WASM
WebGPU
```

---

# 139. Thermal Benchmarking

Especially for:

```text
laptops
Android
iPhone/iPad
MacBooks
```

Measure:

```text
initial throughput
sustained throughput
thermal throttling
battery drain
power
```

---

# 140. Cold Start Optimization

Target:

```text
binary startup
model mapping
metadata parse
backend initialization
weight residency
first token
```

Measure each independently.

---

# 141. Model Startup Strategies

Support:

```text
lazy mapping
lazy tensor residency
preload selected layers
full preload
background warmup
```

---

# 142. Persistent GPU Cache

Persist:

```text
compiled kernels
autotune results
packed layouts
backend-specific plans
```

keyed by:

```text
model hash
GPU identity
driver
backend
kernel version
```

---

# 143. Security / Privacy

Local-first operation must remain a core property.

Provide:

```text
fully offline mode
no telemetry mode
encrypted models
sandboxed server
audit logs
```

---

# 144. Licensing / Provenance

Model and appendix metadata should support:

```text
license
author
source
dataset
modification history
training run
parent model
adapter lineage
```

---

# 145. Documentation

The final documentation set:

```text
Getting Started
Architecture
Format Specification
Compatibility
Hardware
Performance
Backends
CUDA
ROCm
Metal
Vulkan
NPU
Mobile
Server
SDKs
Python
Training
Quantization
Sparsity
Dynamic Growth
Lineage
Self-Training
Research
Security
Distribution
Plugin/Extension Development
```

---

# 146. Extension System

Allow third parties to add:

```text
architecture plugins
backend plugins
tokenizers
quantizers
samplers
model processors
dataset readers
```

without modifying the core.

---

# 147. Stable ABI

Define:

```text
HK C ABI v1
```

with explicit:

```text
version negotiation
struct sizes
capability discovery
error codes
ownership rules
threading rules
device handles
```

---

# 148. Compatibility Contract

Each release must publish:

```text
supported formats
supported architectures
supported backends
supported platforms
known limitations
benchmark suite
ABI version
format version
```

---

# 149. Release Tiers

## Core stable

```text
format
reader
writer
CPU
CUDA
Vulkan
common architectures
server
CLI
```

## Extended stable

```text
ROCm
Metal
mobile
NPU
multimodal
distributed
```

## Experimental

```text
adaptive growth
self-training
self-play
architecture search
advanced sparsity
native training
```

Once research features have sufficient validation, they can graduate.

---

# 150. Project Milestones

## HK 2.0 — Unified Runtime

```text
[ ] backend abstraction
[ ] CUDA engine
[ ] complete CUDA decode
[ ] CUDA prefill
[ ] CUDA attention
[ ] Vulkan integration under common API
[ ] CPU backend finalized
[ ] unified memory planner
[ ] automatic device selection
```

---

## HK 2.1 — Hardware Acceleration

```text
[ ] CUDA Tensor Cores
[ ] 2:4 sparse Tensor Cores
[ ] ROCm/HIP
[ ] Metal
[ ] Vulkan cooperative matrices
[ ] Intel XPU
[ ] DirectML
[ ] mobile GPU paths
```

---

## HK 2.2 — Model Compatibility

```text
[ ] 137+ architecture registry
[ ] Gemma
[ ] Phi
[ ] Mistral variants
[ ] MoE
[ ] Mamba/state-space
[ ] multimodal
[ ] long-context models
```

---

## HK 2.3 — Server Runtime

```text
[ ] continuous batching
[ ] paged KV
[ ] prefix cache
[ ] speculative decoding
[ ] distributed inference
[ ] multi-GPU
```

---

## HK 2.4 — Universal Model Format

```text
[ ] robust v2 format
[ ] checksums
[ ] signatures
[ ] crash-safe mutation
[ ] backend layouts
[ ] external references
[ ] model registry
```

---

## HK 2.5 — Training Platform

```text
[ ] full training
[ ] QLoRA
[ ] LoRA
[ ] distributed training
[ ] distillation
[ ] pruning-aware training
[ ] quantization-aware training
```

---

## HK 2.6 — Adaptive Research Platform

```text
[ ] Net2WiderNet
[ ] Net2DeeperNet
[ ] vocabulary growth
[ ] plasticity isolation
[ ] autonomous growth
[ ] model evolution
[ ] lineage branching/merging
```

---

## HK 2.7 — Self-Improvement Research

```text
[ ] self-conversation
[ ] secure code execution
[ ] self-training
[ ] self-play
[ ] SPIN
[ ] failure mining
[ ] autonomous curriculum
[ ] verified synthetic data
[ ] rollback on regression
```

---

## HK 3.0 — Universal Neural Runtime

Final target:

```text
[ ] desktop
[ ] server
[ ] mobile
[ ] browser
[ ] embedded
[ ] CPU
[ ] CUDA
[ ] ROCm
[ ] Metal
[ ] Vulkan
[ ] DirectML
[ ] Intel XPU
[ ] NPU
[ ] distributed
[ ] multimodal
[ ] training
[ ] model evolution
```

---

# 151. Canonical Dependency Order

Even though everything above is in scope, implementation should follow this dependency chain:

```text
FORMAT
  ↓
MEMORY MAPPING
  ↓
TENSOR / QUANTIZATION CORE
  ↓
CPU KERNELS
  ↓
GRAPH / MODEL ABSTRACTION
  ↓
BACKEND ABSTRACTION
  ↓
CUDA
  ↓
CUDA PREFILL
  ↓
CUDA DECODE
  ↓
CUDA ATTENTION
  ↓
CUDA FUSION / GRAPHS
  ↓
MEMORY PLANNER
  ↓
PARTIAL OFFLOAD
  ↓
VULKAN OPTIMIZATION
  ↓
METAL
  ↓
ROCm
  ↓
NPU BACKENDS
  ↓
MOBILE
  ↓
DISTRIBUTED
  ↓
MODEL ARCHITECTURE EXPANSION
  ↓
SERVER FEATURES
  ↓
ADVANCED QUANTIZATION
  ↓
SPARSITY EXECUTION
  ↓
TRAINING
  ↓
ADAPTIVE GROWTH
  ↓
LINEAGE
  ↓
SELF-TRAINING
  ↓
SELF-PLAY
  ↓
ARCHITECTURE SEARCH
```

---

# 152. Performance Optimization Order

Within every backend:

```text
1. correctness
2. memory traffic
3. data layout
4. vectorization
5. tiling
6. fusion
7. synchronization
8. launch overhead
9. accelerator-specific instructions
10. autotuning
```

Do not jump directly from an unoptimized kernel to complicated accelerator-specific code.

---

# 153. Critical Kernel Priority

The kernel optimization order is:

```text
1. GEMV
2. GEMM
3. attention
4. RMSNorm
5. RoPE
6. SwiGLU
7. KV operations
8. embedding
9. LM head
10. sampling
```

For prefill:

```text
GEMM
attention
fusion
```

For decode:

```text
GEMV
KV bandwidth
attention
launch overhead
```

---

# 154. Memory Optimization Priority

```text
1. mapped weights
2. quantized execution
3. KV segmentation
4. buffer reuse
5. backend residency
6. partial offload
7. KV compression
8. prefix sharing
9. model sharding
10. distributed memory
```

---

# 155. Final Research Program

The research side of HK is considered complete only when it includes:

```text
quantization research
sparsity research
hardware-aware pruning
Net2Net growth
dynamic capacity
plasticity preservation
model lineage
model evolution
self-conversation
secure code evaluation
self-training
self-play
SPIN
distillation
architecture search
hardware-aware architecture search
long-context adaptation
automated optimization
```

Every research feature requires:

```text
theory
implementation
tests
benchmark
ablation
reproducibility
documentation
failure analysis
```

---

# 156. Final Quality Bar

HK is not finished when it "runs a model."

It is finished when:

```text
A model can be downloaded,
converted,
validated,
quantized,
pruned,
sparsified,
edited,
trained,
grown,
versioned,
benchmarked,
served,
distributed,
and executed
```

without leaving the HK ecosystem unless the user chooses to.

And the same logical model should be executable across:

```text
Linux
Windows
macOS
Android
iOS/iPadOS
browser
embedded systems
```

using the best available execution backend.

---

# 157. Final Definition of Success

The final HK project should provide all of the following:

```text
✓ universal neural model container
✓ mmap / zero-copy design
✓ streaming model conversion
✓ GGUF compatibility
✓ SafeTensors compatibility
✓ PyTorch integration
✓ 26+ current storage formats and expanded native formats
✓ common quantization formats
✓ dual-mode quantization
✓ residual quantization
✓ FP8
✓ MXFP4
✓ NVFP4
✓ ternary formats
✓ 2:4 sparsity
✓ BSR / CSR / mask sparsity
✓ CPU SIMD kernels
✓ register-tiled GEMM
✓ register-tiled GEMV
✓ optimized RoPE
✓ optimized softmax
✓ optimized FP8 LUTs
✓ fused operators
✓ CUDA
✓ Tensor Cores
✓ sparse Tensor Cores
✓ ROCm/HIP
✓ Metal
✓ Vulkan
✓ DirectML
✓ Intel XPU
✓ NPU backends
✓ Android
✓ iOS/iPadOS
✓ macOS
✓ Windows
✓ Linux
✓ WebAssembly/WebGPU
✓ long context
✓ paged KV
✓ KV quantization
✓ speculative decoding
✓ continuous batching
✓ prefix caching
✓ multi-GPU
✓ distributed inference
✓ multimodal execution
✓ 137+ architecture target
✓ MoE
✓ state-space models
✓ vision
✓ audio
✓ embeddings
✓ reranking
✓ structured generation
✓ grammar constraints
✓ tool calling
✓ OpenAI-compatible server
✓ native HK protocol
✓ C ABI
✓ C++
✓ Rust
✓ Go
✓ C#
✓ Java
✓ Kotlin
✓ TypeScript
✓ Swift
✓ Python
✓ model hub
✓ model registry
✓ training
✓ QLoRA
✓ LoRA
✓ pruning
✓ distillation
✓ quantization-aware training
✓ dynamic growth
✓ Net2WiderNet
✓ Net2DeeperNet
✓ vocabulary expansion
✓ plasticity isolation
✓ autonomous growth
✓ lineage
✓ cryptographic verification
✓ branching / merging
✓ model deltas
✓ self-conversation
✓ sandboxed code execution
✓ self-training
✓ self-play
✓ SPIN
✓ automated curriculum
✓ failure mining
✓ architecture search
✓ hardware-aware architecture search
✓ unified profiling
✓ autotuning
✓ reproducible benchmarking
✓ security hardening
✓ fuzzing
✓ signed model artifacts
✓ crash-safe editing
```

---

# 158. The Core Strategic Rule

Everything above is in scope.

But the project must always preserve this hierarchy:

```text
                 ┌─────────────────┐
                 │  MODEL QUALITY  │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   CORRECTNESS   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   MEMORY / IO   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │    COMPUTE      │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  ACCELERATION   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   DISTRIBUTED   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   RESEARCH      │
                 └─────────────────┘
```

The research layer must never compromise the correctness and reliability of the runtime.

---

# 159. Final Project Identity

HK should ultimately be understood as four systems sharing one foundation:

```text
                  HK
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
   CONTAINER     RUNTIME     RESEARCH
       │           │           │
       │           │           │
       └───────────┼───────────┘
                   │
                TOOLING
```

### Container

```text
Universal model representation
```

### Runtime

```text
Fast, memory-efficient inference everywhere
```

### Research

```text
Quantization, sparsity, growth, self-training,
self-play, evolution, architecture search
```

### Tooling

```text
Conversion, serving, SDKs, profiling,
benchmarking, packaging, deployment
```

---

# 160. Final North Star

The final question every new feature should answer is:

> **Does this make HK a better universal neural runtime, a better neural model container, a better model-development system, or a better research platform?**

If yes, it belongs in the long-term roadmap.

The final destination is not merely:

```text
"another local LLM runtime"
```

It is:

```text
                  HK
      Universal Neural Model Platform

       ┌────────────┴────────────┐
       │                         │
   UNIVERSAL FORMAT         UNIVERSAL RUNTIME
       │                         │
   model lifecycle          CPU / GPU / NPU
       │                         │
       ├────────────┬────────────┤
       │            │            │
   quantization   sparsity    evolution
       │            │            │
       └────────────┼────────────┘
                    │
               TRAINING
                    │
              SELF-IMPROVEMENT
                    │
             DISTRIBUTED SYSTEM
                    │
             EVERY MAJOR DEVICE
```

**That is the complete HK target.**
