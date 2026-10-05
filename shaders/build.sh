#!/bin/sh
# Recompiles every shader to SPIR-V into src/vk/spv. Needs glslc (the Vulkan SDK or the
# shaderc package). The compiled files are checked in, so ordinary builds do not need glslc.
set -e
cd "$(dirname "$0")"
out=../src/vk/spv
mkdir -p "$out"
for t in f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl q2_k q3_k q4_k q5_k q6_k iq4_xs; do
    T=$(echo "$t" | tr a-z A-Z)
    for k in mmv mm embed; do
        glslc --target-env=vulkan1.2 -O -I. -DTYPE_"$T" "$k.comp" -o "$out/${k}_$t.spv"
    done
done
for k in rmsnorm rope_kv attention swiglu swiglu_pair add vadd; do
    glslc --target-env=vulkan1.2 -O -I. "$k.comp" -o "$out/$k.spv"
done
