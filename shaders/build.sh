#!/bin/sh
# Recompiles every shader to SPIR-V into src/vk/spv. Needs glslc (the Vulkan SDK or the
# shaderc package). The compiled files are checked in, so ordinary builds do not need glslc.
set -e
cd "$(dirname "$0")"
out=../src/vk/spv
mkdir -p "$out"
for t in f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl q2_k q3_k q4_k q5_k q6_k iq4_xs; do
    T=$(echo "$t" | tr a-z A-Z)
    for k in mmv mm mm_cm embed; do
        glslc --target-env=vulkan1.2 -O -I. -DTYPE_"$T" "$k.comp" -o "$out/${k}_$t.spv"
    done
done
for k in rmsnorm rope_kv attention attention_part attention_combine swiglu swiglu_pair add vadd mm_sparse_2_4 mm_bsr flash_attention; do
    glslc --target-env=vulkan1.2 -O -I. "$k.comp" -o "$out/$k.spv"
done
# Cooperative matrix prompt attention, one binary per head size it supports.
for hd in 64 128; do
    glslc --target-env=vulkan1.2 -O -I. -DHD="$hd" attention_cm.comp -o "$out/attention_cm_$hd.spv"
done
# Integer dot product decode kernels (VK_KHR_shader_integer_dot_product) and the activation quantizer they read.
for t in q8_0 q4_0 q4_1 q5_0 q5_1 q4_k q5_k q6_k; do
    T=$(echo "$t" | tr a-z A-Z)
    glslc --target-env=vulkan1.2 -O -I. -DTYPE_"$T" mmvq.comp -o "$out/mmvq_$t.spv"
done
glslc --target-env=vulkan1.2 -O -I. qact.comp -o "$out/qact.spv"
