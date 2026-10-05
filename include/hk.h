#ifndef HK_H
#define HK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef HK_BUILDING_DLL
    #define HK_API __declspec(dllexport)
  #else
    #define HK_API __declspec(dllimport)
  #endif
#else
  #define HK_API __attribute__((visibility("default")))
#endif

typedef enum {
    HK_STORAGE_F32 = 0x00,
    HK_STORAGE_F16 = 0x01,
    HK_STORAGE_BF16 = 0x02,
    HK_STORAGE_FP8_E4M3 = 0x03,
    HK_STORAGE_FP8_E5M2 = 0x04,
    HK_STORAGE_INT8 = 0x05,
    HK_STORAGE_INT32 = 0x06,
    HK_STORAGE_INT64 = 0x07,
    HK_STORAGE_UINT8 = 0x08,
    HK_STORAGE_BOOL = 0x09,
    HK_STORAGE_INT16 = 0x0A,
    HK_STORAGE_UINT16 = 0x0B,
    HK_STORAGE_UINT32 = 0x0C,
    HK_STORAGE_UINT64 = 0x0D,
    HK_STORAGE_F64 = 0x0E,
    HK_STORAGE_DQ4 = 0x10,
    HK_STORAGE_DQ8 = 0x11,
    HK_STORAGE_DQ6 = 0x12,
    HK_STORAGE_DQ12 = 0x13,
    HK_STORAGE_DQT = 0x14,
    HK_STORAGE_Q4_0 = 0x15,
    HK_STORAGE_Q8_0 = 0x16,
    HK_STORAGE_Q4_1 = 0x17,
    HK_STORAGE_Q5_0 = 0x18,
    HK_STORAGE_Q5_1 = 0x19,
    HK_STORAGE_Q8_1 = 0x1A,
    HK_STORAGE_SPARSE_F16 = 0x20,
    HK_STORAGE_SPARSE_DQ8 = 0x21,
    HK_STORAGE_SPARSE_2_4 = 0x22,
    HK_STORAGE_SPARSE_DQ4_2_4 = 0x23,
    HK_STORAGE_NULL_REF = 0x30,
    HK_STORAGE_SHARED_REF = 0x31,
    HK_STORAGE_LORA_REF = 0x32,

    // Advanced K-Quants (256-element super-blocks)
    HK_STORAGE_Q2_K = 0x40,
    HK_STORAGE_Q3_K = 0x41,
    HK_STORAGE_Q4_K = 0x42,
    HK_STORAGE_Q5_K = 0x43,
    HK_STORAGE_Q6_K = 0x44,
    HK_STORAGE_Q8_K = 0x45,

    // Advanced I-Quants
    HK_STORAGE_IQ1_S = 0x50,
    HK_STORAGE_IQ1_M = 0x51,
    HK_STORAGE_IQ2_XXS = 0x52,
    HK_STORAGE_IQ2_XS = 0x53,
    HK_STORAGE_IQ3_XXS = 0x54,
    HK_STORAGE_IQ4_NL = 0x55,
    HK_STORAGE_IQ4_XS = 0x56,
    HK_STORAGE_IQ2_S = 0x57,
    HK_STORAGE_IQ3_S = 0x58,

    // Microscaling & Ternary Formats
    HK_STORAGE_TQ1_0 = 0x60,
    HK_STORAGE_TQ2_0 = 0x61,
    HK_STORAGE_MXFP4 = 0x62,
    HK_STORAGE_NVFP4 = 0x63
} hk_storage_type_t;

typedef enum {
    HK_TILE_ROW_MAJOR = 0x00,
    HK_TILE_COL_MAJOR = 0x01,
    HK_TILE_16X16 = 0x02,
    HK_TILE_16X8 = 0x03,
    HK_TILE_32X16 = 0x04,
    HK_TILE_BLOCK_SPARSE_2_4 = 0x05,
    HK_TILE_32X32 = 0x06,
    HK_TILE_64X64 = 0x07
} hk_tile_layout_t;

typedef enum {
    HK_SPARSITY_NONE = 0x00,
    HK_SPARSITY_BITMASK = 0x01,
    HK_SPARSITY_CSR = 0x02,
    HK_SPARSITY_STRUCTURED_2_4 = 0x03,
    HK_SPARSITY_PHYSICAL_PRUNED = 0x04,
    HK_SPARSITY_BSR = 0x05
} hk_sparsity_type_t;

typedef struct {
    const char* name;
    uint8_t storage_type;
    uint8_t tile_layout;
    uint8_t sparsity_type;
    uint8_t ndim;
    uint64_t shape[8];
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t residual_offset;
    uint64_t residual_size;
    uint64_t scale_offset;
    uint64_t scale_size;
    uint16_t block_size;
    float sparsity_ratio;
} hk_tensor_info_t;

typedef struct hk_reader_t hk_reader_t;

// Lifecycle & File Access
HK_API hk_reader_t* hk_open(const char* path);
HK_API void hk_close(hk_reader_t* reader);

// Tensor Access
HK_API uint64_t hk_get_tensor_count(const hk_reader_t* reader);
HK_API int hk_get_tensor_info(const hk_reader_t* reader, uint64_t index, hk_tensor_info_t* out_info);
HK_API uint64_t hk_get_all_tensor_infos(const hk_reader_t* reader, hk_tensor_info_t* out_infos, uint64_t max_count);
HK_API const void* hk_get_tensor_data(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);
HK_API const void* hk_get_tensor_residual(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);
HK_API const void* hk_get_tensor_scales(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);

// Dequantization & Precision Recovery
HK_API int hk_dequantize_f32(
    const hk_reader_t* reader,
    uint64_t index,
    int with_residual,
    float* out_buf,
    uint64_t count
);

// Metadata
HK_API const char* hk_get_metadata_string(const hk_reader_t* reader, const char* key);
HK_API int hk_get_metadata_int(const hk_reader_t* reader, const char* key, int64_t* out_val);
HK_API int hk_get_metadata_float(const hk_reader_t* reader, const char* key, double* out_val);
HK_API int hk_get_metadata_bool(const hk_reader_t* reader, const char* key, int* out_val);

// Sharding & Multi-File Container Support
#define HK_FLAG_IS_SHARDED 0x40

HK_API int hk_reader_is_sharded(const hk_reader_t* reader);
HK_API uint16_t hk_reader_get_split_index(const hk_reader_t* reader);
HK_API uint16_t hk_reader_get_split_count(const hk_reader_t* reader);

// In-Place Key-Value Metadata Patching
HK_API int hk_metadata_patch_in_place(const char* file_path, const char* key, const char* val);

// Low-level Quantization Helpers
HK_API float hk_quantize_block_nf4(const float* block, uint32_t count, uint8_t* packed_out, float* residual_out);
HK_API float hk_quantize_block_dq8(const float* block, uint32_t count, int8_t* out_i8, float* residual_out);
HK_API float hk_quantize_block_dqt(const float* block, uint32_t count, uint8_t* packed_out, float* residual_out);

// Sparsity & Tiling Helpers
HK_API int hk_pack_2_4(const float* dense_in, uint64_t count, uint8_t* out_bytes);
HK_API int hk_unpack_2_4(const uint8_t* payload, uint64_t payload_len, uint64_t count, float* out_buf);
HK_API int hk_tile_16x16_pack(const float* in_row_major, uint64_t M, uint64_t K, float* out_tiled);
HK_API int hk_tile_16x16_unpack(const float* in_tiled, uint64_t M, uint64_t K, float* out_row_major);

// Appendix Definitions
typedef enum {
    HK_APPENDIX_LORA_ADAPTER = 0x01,
    HK_APPENDIX_DELTA_PATCH = 0x02,
    HK_APPENDIX_NEW_LAYER = 0x03,
    HK_APPENDIX_CODE_EVAL = 0x04,
    HK_APPENDIX_KV_CACHE_SINK = 0x05,
    HK_APPENDIX_TOPOLOGY_HEAD = 0x06
} hk_appendix_type_t;

typedef struct {
    uint8_t entry_type;
    uint8_t flags;
    uint32_t generation;
    uint64_t timestamp;
    uint8_t parent_hash[32];
    float metric_loss;
    float metric_acc;
    float metric_pass;
    float metric_custom;
    const char* name;
    const char* target;
    const void* data;
    uint64_t data_size;
} hk_appendix_entry_t;

// Appendix Access & Mutation
HK_API uint64_t hk_appendix_get_count(const hk_reader_t* reader);
HK_API int hk_appendix_get_entry(const hk_reader_t* reader, uint64_t index, hk_appendix_entry_t* out_entry);
HK_API int hk_appendix_append(
    const char* file_path,
    uint8_t entry_type,
    uint8_t flags,
    const char* name,
    const char* target,
    uint32_t generation,
    const uint8_t parent_hash[32],
    float metric_loss,
    float metric_acc,
    float metric_pass,
    float metric_custom,
    const void* data,
    uint64_t data_size
);
HK_API int hk_appendix_rollback(const char* file_path, uint32_t target_generation);

// Native SIMD Tensor Operations & Inference Kernels
HK_API float hk_dot_product_f32(const float* a, const float* b, uint64_t count);
HK_API void hk_gemv_f32(
    const float* W,
    const float* x,
    const float* bias,
    float* y,
    uint64_t M,
    uint64_t K
);
HK_API void hk_gemm_f32(
    const float* A,
    const float* B,
    float* C,
    uint64_t M,
    uint64_t K,
    uint64_t N
);
HK_API void hk_fused_gemv_nf4(
    const uint8_t* packed_W,
    const float* scales,
    const float* x,
    const float* bias,
    float* y,
    uint64_t M,
    uint64_t K,
    uint32_t block_size
);
HK_API void hk_fused_gemv_dq8(
    const int8_t* W_i8,
    const float* scales,
    const float* x,
    const float* bias,
    float* y,
    uint64_t M,
    uint64_t K,
    uint32_t block_size
);

HK_API void hk_gemv_bf16(
    const uint16_t* W_bf16,
    const float* x,
    const float* bias,
    float* y,
    size_t M,
    size_t K
);
HK_API void hk_gemv_f16(
    const void* W_f16,
    const float* x,
    const float* bias,
    float* y,
    size_t M,
    size_t K
);
HK_API void hk_gemv_int8(
    const int8_t* W_i8,
    const float* x,
    float scale_w,
    const float* bias,
    float* y,
    size_t M,
    size_t K
);

// Native Dynamic Architecture Growth (Net2Net)
HK_API int hk_net2wider(
    const float* w_in_old,
    const float* b_in_old,
    float* w_in_new,
    float* b_in_new,
    const float* w_out_old,
    float* w_out_new,
    uint64_t old_out,
    uint64_t new_out,
    uint64_t in_f,
    uint64_t out_f,
    float noise_std,
    uint64_t seed
);
HK_API void hk_net2deeper(float* weights, float* bias, uint64_t dim);

HK_API int hk_net2wider_swiglu(
    const float* w_gate_old,
    float* w_gate_new,
    const float* b_gate_old,
    float* b_gate_new,
    const float* w_up_old,
    float* w_up_new,
    const float* b_up_old,
    float* b_up_new,
    const float* w_down_old,
    float* w_down_new,
    const float* b_down_old,
    float* b_down_new,
    uint64_t old_inter,
    uint64_t new_inter,
    uint64_t in_features,
    uint64_t out_features,
    int zero_init,
    float noise_std,
    uint64_t seed
);

HK_API int hk_expand_vocab(
    const float* embed_old,
    float* embed_new,
    const float* lm_head_old,
    float* lm_head_new,
    uint64_t old_vocab,
    uint64_t new_vocab,
    uint64_t hidden_dim,
    uint64_t seed
);

HK_API void hk_plasticity_mask_rows(float* grad, uint64_t total_elements, uint64_t cutoff_rows, uint64_t cols);
HK_API void hk_plasticity_mask_cols(float* grad, uint64_t total_elements, uint64_t rows, uint64_t cutoff_cols, uint64_t cols);

// SIMD Vector Activations & Normalizations
HK_API int hk_forward_swiglu(
    const float* x,
    const float* w_gate,
    const float* b_gate,
    const float* w_up,
    const float* b_up,
    const float* w_down,
    const float* b_down,
    float* intermediate_buf,
    float* out,
    uint64_t in_features,
    uint64_t inter_features,
    uint64_t out_features
);
HK_API void hk_forward_rmsnorm(const float* x, const float* weight, float eps, float* out, uint64_t n);
HK_API void hk_forward_silu(const float* x, float* out, uint64_t n);

// Container Writer API
typedef struct hk_writer_t hk_writer_t;

HK_API hk_writer_t* hk_writer_create(uint64_t alignment);
HK_API void hk_writer_destroy(hk_writer_t* writer);
HK_API void hk_writer_set_sharding(hk_writer_t* writer, uint16_t split_index, uint16_t split_count);
HK_API void hk_writer_set_raw_storage(hk_writer_t* writer, int enabled);
HK_API int hk_writer_add_metadata_string(hk_writer_t* writer, const char* key, const char* val);
HK_API int hk_writer_add_metadata_int(hk_writer_t* writer, const char* key, int64_t val);
HK_API int hk_writer_add_metadata_float(hk_writer_t* writer, const char* key, double val);
HK_API int hk_writer_add_metadata_bool(hk_writer_t* writer, const char* key, int val);
HK_API int hk_writer_add_metadata_json(hk_writer_t* writer, const char* key, const char* json);
HK_API int hk_writer_add_tensor(
    hk_writer_t* writer,
    const char* name,
    uint8_t storage_type,
    uint8_t tile_layout,
    uint8_t sparsity_type,
    uint8_t ndim,
    const uint64_t* shape,
    const uint8_t* data,
    uint64_t data_len,
    float sparsity_ratio
);
HK_API int hk_writer_write_to_file(hk_writer_t* writer, const char* path);

// Advanced Quantization & Microscaling C ABI
HK_API int hk_quantize_block_q4_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q4_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q8_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q8_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_q6_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_q2_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_iq4_nl(const uint8_t* packed_in, float scale, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_mxfp4(const uint8_t* packed_in, uint8_t scale_e8m0, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_nvfp4(const uint8_t* packed_in, uint8_t scale_fp8, uint32_t count, float* out_f32);

// GGUF Quantization & Dequantization C ABI
HK_API int hk_quantize_block_q4_0(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q4_0(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q8_0(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q8_0(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q5_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q5_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q3_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q3_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q6_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_quantize_block_q2_k(const float* weights, uint32_t count, void* out_block);

// Whole-tensor quantization. `count` must be a multiple of the format's block size
// (32 for q4_0/q8_0, 256 for the K formats). Output buffers are sized by the caller.
HK_API int hk_quantize_tensor_q4_0(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q4_0(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q8_0(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q8_0(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q2_k(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q2_k(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q3_k(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q3_k(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q4_k(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q4_k(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q5_k(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q5_k(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q6_k(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q6_k(const uint8_t* in_bytes, uint64_t count, float* out_f32);
HK_API int hk_quantize_tensor_q8_k(const float* weights, uint64_t count, uint8_t* out_bytes);
HK_API int hk_dequantize_tensor_q8_k(const uint8_t* in_bytes, uint64_t count, float* out_f32);
// NF4 with per-block f32 scales and an optional residual (pass NULL to skip).
HK_API int hk_quantize_tensor_nf4(const float* weights, uint64_t count, uint32_t block_size, uint8_t* packed_out, float* scales_out, float* residual_out);
HK_API int hk_dequantize_tensor_nf4(const uint8_t* packed_in, const float* scales_in, uint64_t count, uint32_t block_size, float* out_f32, const float* residual_in);

// Packed-Weight SIMD GEMV Kernels
HK_API int hk_gemv_q8_0(const void* w_packed, const float* x, const float* bias, float* out, uint64_t m, uint64_t k);
HK_API int hk_gemv_q4_0(const void* w_packed, const float* x, const float* bias, float* out, uint64_t m, uint64_t k);
HK_API int hk_gemv_q4_k(const void* w_packed, const float* x, const float* bias, float* out, uint64_t m, uint64_t k);

// Mathematical Tensor Transformations
HK_API int hk_rope_permute_hf_to_gguf(const float* in_w, float* out_w, uint64_t n_heads, uint64_t head_dim, uint64_t batch_size);
HK_API int hk_rope_unpermute_gguf_to_hf(const float* in_w, float* out_w, uint64_t n_heads, uint64_t head_dim, uint64_t batch_size);
// Adds `offset` to `len` values in place (Gemma style norm weight offset; pass -1 to undo).
HK_API void hk_layernorm_offset_f32(float* data, uint64_t len, float offset);

// GGUF Transcoder C ABI
HK_API int hk_convert_gguf(const char* in_gguf_path, const char* out_hk_path);
HK_API int hk_export_gguf(const char* in_hk_path, const char* out_gguf_path);
// Converts a .safetensors file to .hk, storing floats as `storage_type` (an hk_storage_type_t).
HK_API int hk_convert_safetensors(const char* input_path, const char* output_path, uint8_t storage_type);

// Hugging Face Architecture Mapper C ABI
HK_API int hk_hf_detect_architecture(const char* json_config, char* out_arch, size_t max_len);
HK_API int hk_hf_map_tensor_name(const char* tensor_name, const char* arch, int to_hk, char* out_name, size_t max_len);

// Native Context Window Management C ABI
HK_API int hk_context_truncate(
    const uint32_t* in_tokens,
    size_t in_len,
    size_t max_tokens,
    int strategy,
    float head_ratio,
    uint32_t* out_tokens,
    size_t* out_len
);

// Native Adaptive Autonomous Framework C ABI
HK_API int hk_governor_can_grow(
    uint64_t current_params,
    uint64_t added_params,
    float max_growth_ratio,
    uint64_t max_vram_mb,
    uint32_t dtype_bytes,
    char* out_reason,
    size_t max_reason_len
);

HK_API int hk_governor_can_grow_batch(
    const uint64_t* current_params,
    const uint64_t* added_params,
    size_t n,
    float max_growth_ratio,
    uint64_t max_vram_mb,
    uint32_t dtype_bytes,
    uint8_t* out_results
);

HK_API int hk_expand_vocab_embeddings(
    const float* old_embed,
    size_t old_vocab,
    size_t hidden_size,
    size_t new_vocab,
    float* new_embed,
    float init_std,
    uint64_t seed
);

HK_API int hk_init_plasticity_mask(
    float* mask,
    size_t total_units,
    size_t base_units,
    float decay_rate
);

// Multi-Device Alignment Constants & Flags
#define HK_DEFAULT_ALIGNMENT_BYTES 128
#define HK_UNIVERSAL_PAGE_ALIGNMENT_BYTES 4096
#define HK_APPLE_SILICON_ALIGNMENT_BYTES 16384
#define HK_DIRECT_DMA_ALIGNMENT_BYTES 65536

#define HK_FLAG_RAW_WEIGHT_STORAGE (1 << 7)
#define HK_FLAG_UNIVERSAL_PAGE_ALIGNED (1 << 8)

typedef struct {
    uint8_t vendor;
    uint8_t has_avx2;
    uint8_t has_avx512f;
    uint8_t has_avx512vnni;
    uint8_t has_avx_vnni;
    uint8_t has_amx;
    uint8_t has_arm_neon;
    uint8_t has_arm_sve;
    uint8_t is_apple_silicon;
    uint8_t has_rocm_ready;
    uint8_t has_npu_ready;
    uint8_t reserved[5];
    uint64_t optimal_page_alignment;
    uint64_t dma_hugepage_alignment;
} hk_hardware_caps_t;

// Hardware Profiling & Zero-Copy Universal Alignment
HK_API void hk_detect_hardware(hk_hardware_caps_t* out_caps);
HK_API size_t hk_get_optimal_alignment(void);
HK_API int hk_is_raw_storage(const hk_reader_t* reader);
HK_API int hk_is_universal_page_aligned(const hk_reader_t* reader);
HK_API uint32_t hk_get_file_alignment(const hk_reader_t* reader);
HK_API const void* hk_get_tensor_raw_ptr(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);
HK_API const void* hk_get_raw_buffer(const hk_reader_t* reader, uint64_t* out_size);

// Raw Weights Linear Algebra (Zero-Copy GEMV & Dot Products)
HK_API void hk_gemv_bf16(const uint16_t* w_bf16, const float* x, const float* bias, float* y, size_t m, size_t k);
HK_API void hk_gemv_f16(const void* w_f16, const float* x, const float* bias, float* y, size_t m, size_t k);
HK_API void hk_gemv_int8(const int8_t* w_i8, const float* x, float scale_w, const float* bias, float* y, size_t m, size_t k);
HK_API float hk_dot_bf16(const uint16_t* a, const float* b, size_t len);
HK_API float hk_dot_f16(const void* a, const float* b, size_t len);
HK_API int32_t hk_dot_int8(const int8_t* a, const int8_t* b, size_t len);

// Tokenizer. Reads the vocabulary stored in the container.
// Not thread safe: use one tokenizer per thread, or serialise calls.
typedef struct hk_tokenizer_t hk_tokenizer_t;
HK_API hk_tokenizer_t* hk_tokenizer_load_from_file(const char* path);
HK_API void hk_tokenizer_free(hk_tokenizer_t* tok);
HK_API uint32_t hk_tokenizer_get_vocab_size(const hk_tokenizer_t* tok);
// Returns the number of ids written, or 0 on failure. add_special adds BOS/EOS where the model
// wants them. parse_special turns text such as "<|im_start|>" into control token ids.
HK_API uint32_t hk_tokenizer_encode(hk_tokenizer_t* tok, const char* text, int add_special, int parse_special, uint32_t* out_ids, uint32_t max_ids);
// Returns the number of bytes written (not NUL terminated).
HK_API uint32_t hk_tokenizer_decode(const hk_tokenizer_t* tok, const uint32_t* ids, uint32_t num_ids, int show_special, uint8_t* out_buf, uint32_t max_len);

// Inference engine (Llama, Qwen2 and Qwen3 style dense models). Weights are memory mapped.
typedef struct hk_engine_t hk_engine_t;
// Returns NULL on failure. hk_engine_last_error then holds the reason (not thread safe).
HK_API hk_engine_t* hk_engine_load_from_file(const char* path);
HK_API uint32_t hk_engine_last_error(char* out, uint32_t capacity);
HK_API void hk_engine_free(hk_engine_t* engine);
HK_API uint32_t hk_engine_get_vocab_size(const hk_engine_t* engine);
HK_API uint32_t hk_engine_get_context_size(const hk_engine_t* engine);
// Forget the conversation. The next call must start at position 0.
HK_API void hk_engine_reset_cache(hk_engine_t* engine);
// Feeds n tokens starting at position pos and writes the logits of the last one to out_logits
// (vocabulary sized). Returns 0 on success, -1 for a null argument, -2 when the context window is
// full, -3 when n exceeds the batch limit, -4 on any other failure.
HK_API int hk_engine_forward_tokens(hk_engine_t* engine, const uint32_t* tokens, uint32_t n, uint32_t pos, float* out_logits);
HK_API int hk_engine_forward(hk_engine_t* engine, uint32_t token, uint32_t pos, float* out_logits);

// Sampling. logits is modified in place. temp 0 is greedy. seed 0 uses the clock.
HK_API uint32_t hk_sample_token(float* logits, uint64_t vocab_size, float temp, uint32_t top_k, float top_p, float min_p, float repeat_penalty, const uint32_t* history, uint32_t history_len, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif // HK_H
