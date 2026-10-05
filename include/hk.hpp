/**
 * HK C++20 Modern Header-Only Interface
 * Provides RAII Model, Tensor abstractions, SIMD inference acceleration,
 * dynamic architecture growth (Net2Wider / SwiGLU / Vocab), and container writing.
 */

#ifndef HK_HPP
#define HK_HPP

#include "hk.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <optional>
#include <string_view>
#include <span>
#include <iostream>

namespace hk {

enum class StorageType : uint8_t {
    F32 = HK_STORAGE_F32,
    F16 = HK_STORAGE_F16,
    BF16 = HK_STORAGE_BF16,
    FP8_E4M3 = HK_STORAGE_FP8_E4M3,
    FP8_E5M2 = HK_STORAGE_FP8_E5M2,
    INT8 = HK_STORAGE_INT8,
    INT32 = HK_STORAGE_INT32,
    INT64 = HK_STORAGE_INT64,
    UINT8 = HK_STORAGE_UINT8,
    BOOL = HK_STORAGE_BOOL,
    INT16 = HK_STORAGE_INT16,
    UINT16 = HK_STORAGE_UINT16,
    UINT32 = HK_STORAGE_UINT32,
    UINT64 = HK_STORAGE_UINT64,
    F64 = HK_STORAGE_F64,
    DQ4 = HK_STORAGE_DQ4,
    NF4 = HK_STORAGE_DQ4,
    DQ8 = HK_STORAGE_DQ8,
    DQ6 = HK_STORAGE_DQ6,
    DQ12 = HK_STORAGE_DQ12,
    DQT = HK_STORAGE_DQT,
    Q4_0 = HK_STORAGE_Q4_0,
    Q8_0 = HK_STORAGE_Q8_0,
    Q4_1 = HK_STORAGE_Q4_1,
    Q5_0 = HK_STORAGE_Q5_0,
    Q5_1 = HK_STORAGE_Q5_1,
    Q8_1 = HK_STORAGE_Q8_1,
    SparseF16 = HK_STORAGE_SPARSE_F16,
    SparseDQ8 = HK_STORAGE_SPARSE_DQ8,
    Sparse24 = HK_STORAGE_SPARSE_2_4,
    SparseDQ4_2_4 = HK_STORAGE_SPARSE_DQ4_2_4,
    NullRef = HK_STORAGE_NULL_REF,
    SharedRef = HK_STORAGE_SHARED_REF,
    LoRARef = HK_STORAGE_LORA_REF,
    Q2_K = HK_STORAGE_Q2_K,
    Q3_K = HK_STORAGE_Q3_K,
    Q4_K = HK_STORAGE_Q4_K,
    Q5_K = HK_STORAGE_Q5_K,
    Q6_K = HK_STORAGE_Q6_K,
    Q8_K = HK_STORAGE_Q8_K,
    IQ1_S = HK_STORAGE_IQ1_S,
    IQ1_M = HK_STORAGE_IQ1_M,
    IQ2_XXS = HK_STORAGE_IQ2_XXS,
    IQ2_XS = HK_STORAGE_IQ2_XS,
    IQ3_XXS = HK_STORAGE_IQ3_XXS,
    IQ4_NL = HK_STORAGE_IQ4_NL,
    IQ4_XS = HK_STORAGE_IQ4_XS,
    IQ2_S = HK_STORAGE_IQ2_S,
    IQ3_S = HK_STORAGE_IQ3_S,
    TQ1_0 = HK_STORAGE_TQ1_0,
    TQ2_0 = HK_STORAGE_TQ2_0,
    MXFP4 = HK_STORAGE_MXFP4,
    NVFP4 = HK_STORAGE_NVFP4
};

enum class TileLayout : uint8_t {
    RowMajor = HK_TILE_ROW_MAJOR,
    ColMajor = HK_TILE_COL_MAJOR,
    Tile16x16 = HK_TILE_16X16,
    Tile16x8 = HK_TILE_16X8,
    Tile32x16 = HK_TILE_32X16,
    BlockSparse24 = HK_TILE_BLOCK_SPARSE_2_4,
    Tile32x32 = HK_TILE_32X32,
    Tile64x64 = HK_TILE_64X64
};

enum class SparsityType : uint8_t {
    None = HK_SPARSITY_NONE,
    Bitmask = HK_SPARSITY_BITMASK,
    CSR = HK_SPARSITY_CSR,
    Structured24 = HK_SPARSITY_STRUCTURED_2_4,
    PhysicalPruned = HK_SPARSITY_PHYSICAL_PRUNED,
    BSR = HK_SPARSITY_BSR
};

enum class AppendixType : uint8_t {
    LoRAAdapter = HK_APPENDIX_LORA_ADAPTER,
    DeltaPatch = HK_APPENDIX_DELTA_PATCH,
    NewLayer = HK_APPENDIX_NEW_LAYER,
    CodeEval = HK_APPENDIX_CODE_EVAL,
    KVCacheSink = HK_APPENDIX_KV_CACHE_SINK,
    TopologyHead = HK_APPENDIX_TOPOLOGY_HEAD
};

class Tensor {
public:
    Tensor(const hk_reader_t* reader, uint64_t index, const hk_tensor_info_t& info)
        : m_reader(reader), m_index(index), m_info(info) {}

    std::string_view name() const { return m_info.name ? m_info.name : ""; }
    StorageType storage_type() const { return static_cast<StorageType>(m_info.storage_type); }
    TileLayout tile_layout() const { return static_cast<TileLayout>(m_info.tile_layout); }
    SparsityType sparsity_type() const { return static_cast<SparsityType>(m_info.sparsity_type); }
    uint8_t ndim() const { return m_info.ndim; }
    uint16_t block_size() const { return m_info.block_size; }
    float sparsity_ratio() const { return m_info.sparsity_ratio; }
    
    std::vector<uint64_t> shape() const {
        return std::vector<uint64_t>(m_info.shape, m_info.shape + m_info.ndim);
    }

    uint64_t element_count() const {
        uint64_t cnt = 1;
        for (uint8_t i = 0; i < m_info.ndim; ++i) {
            cnt *= m_info.shape[i];
        }
        return cnt;
    }

    std::span<const uint8_t> raw_data() const {
        uint64_t size = 0;
        const void* ptr = hk_get_tensor_data(m_reader, m_index, &size);
        if (!ptr || size == 0) return {};
        return std::span<const uint8_t>(static_cast<const uint8_t*>(ptr), size);
    }

    std::span<const uint8_t> raw_residual() const {
        uint64_t size = 0;
        const void* ptr = hk_get_tensor_residual(m_reader, m_index, &size);
        if (!ptr || size == 0) return {};
        return std::span<const uint8_t>(static_cast<const uint8_t*>(ptr), size);
    }

    std::span<const uint8_t> raw_scales() const {
        uint64_t size = 0;
        const void* ptr = hk_get_tensor_scales(m_reader, m_index, &size);
        if (!ptr || size == 0) return {};
        return std::span<const uint8_t>(static_cast<const uint8_t*>(ptr), size);
    }

    std::vector<float> dequantize(bool with_residual = true) const {
        uint64_t count = element_count();
        std::vector<float> buffer(count);
        int res = hk_dequantize_f32(m_reader, m_index, with_residual ? 1 : 0, buffer.data(), count);
        if (res != 0) {
            throw std::runtime_error("Failed to dequantize tensor: " + std::string(name()));
        }
        return buffer;
    }

private:
    const hk_reader_t* m_reader;
    uint64_t m_index;
    hk_tensor_info_t m_info;
};

class Model {
public:
    static std::shared_ptr<Model> open(const std::string& path) {
        hk_reader_t* r = hk_open(path.c_str());
        if (!r) {
            throw std::runtime_error("Failed to open HK model file: " + path);
        }
        return std::shared_ptr<Model>(new Model(r));
    }

    ~Model() {
        if (m_reader) {
            hk_close(m_reader);
            m_reader = nullptr;
        }
    }

    // Move only
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept : m_reader(other.m_reader) { other.m_reader = nullptr; }
    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            if (m_reader) hk_close(m_reader);
            m_reader = other.m_reader;
            other.m_reader = nullptr;
        }
        return *this;
    }

    uint64_t tensor_count() const {
        return hk_get_tensor_count(m_reader);
    }

    Tensor get_tensor(uint64_t index) const {
        hk_tensor_info_t info;
        if (hk_get_tensor_info(m_reader, index, &info) != 0) {
            throw std::out_of_range("Tensor index out of range: " + std::to_string(index));
        }
        return Tensor(m_reader, index, info);
    }

    std::optional<Tensor> get_tensor(const std::string& name) const {
        uint64_t count = tensor_count();
        for (uint64_t i = 0; i < count; ++i) {
            hk_tensor_info_t info;
            if (hk_get_tensor_info(m_reader, i, &info) == 0) {
                if (info.name && name == info.name) {
                    return Tensor(m_reader, i, info);
                }
            }
        }
        return std::nullopt;
    }

    std::optional<std::string> get_metadata_string(const std::string& key) const {
        const char* val = hk_get_metadata_string(m_reader, key.c_str());
        if (val) return std::string(val);
        return std::nullopt;
    }

    std::optional<int64_t> get_metadata_int(const std::string& key) const {
        int64_t val = 0;
        if (hk_get_metadata_int(m_reader, key.c_str(), &val) == 0) {
            return val;
        }
        return std::nullopt;
    }

    std::optional<double> get_metadata_float(const std::string& key) const {
        double val = 0.0;
        if (hk_get_metadata_float(m_reader, key.c_str(), &val) == 0) {
            return val;
        }
        return std::nullopt;
    }

    std::optional<bool> get_metadata_bool(const std::string& key) const {
        int val = 0;
        if (hk_get_metadata_bool(m_reader, key.c_str(), &val) == 0) {
            return val != 0;
        }
        return std::nullopt;
    }

    uint64_t appendix_count() const {
        return hk_appendix_get_count(m_reader);
    }

    std::optional<hk_appendix_entry_t> get_appendix_entry(uint64_t index) const {
        hk_appendix_entry_t entry;
        if (hk_appendix_get_entry(m_reader, index, &entry) == 0) {
            return entry;
        }
        return std::nullopt;
    }

    bool is_sharded() const {
        return hk_reader_is_sharded(m_reader) != 0;
    }

    uint16_t split_index() const {
        return hk_reader_get_split_index(m_reader);
    }

    uint16_t split_count() const {
        return hk_reader_get_split_count(m_reader);
    }

    static bool patch_metadata_in_place(const std::string& path, const std::string& key, const std::string& val) {
        return hk_metadata_patch_in_place(path.c_str(), key.c_str(), val.c_str()) == 0;
    }

private:
    explicit Model(hk_reader_t* reader) : m_reader(reader) {}
    hk_reader_t* m_reader;
};

class Writer {
public:
    explicit Writer(uint64_t alignment = 64) {
        m_writer = hk_writer_create(alignment);
        if (!m_writer) {
            throw std::runtime_error("Failed to create HK writer");
        }
    }

    ~Writer() {
        if (m_writer) {
            hk_writer_destroy(m_writer);
            m_writer = nullptr;
        }
    }

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    Writer(Writer&& other) noexcept : m_writer(other.m_writer) {
        other.m_writer = nullptr;
    }

    Writer& operator=(Writer&& other) noexcept {
        if (this != &other) {
            if (m_writer) hk_writer_destroy(m_writer);
            m_writer = other.m_writer;
            other.m_writer = nullptr;
        }
        return *this;
    }

    void set_sharding(uint16_t split_index, uint16_t split_count) {
        hk_writer_set_sharding(m_writer, split_index, split_count);
    }

    void add_metadata(const std::string& key, const std::string& val) {
        if (hk_writer_add_metadata_string(m_writer, key.c_str(), val.c_str()) != 0) {
            throw std::runtime_error("Failed to add string metadata: " + key);
        }
    }

    // Without this, a string literal would pick the bool overload.
    void add_metadata(const std::string& key, const char* val) {
        add_metadata(key, std::string(val));
    }

    void add_metadata(const std::string& key, int64_t val) {
        if (hk_writer_add_metadata_int(m_writer, key.c_str(), val) != 0) {
            throw std::runtime_error("Failed to add int metadata: " + key);
        }
    }

    void add_metadata(const std::string& key, double val) {
        if (hk_writer_add_metadata_float(m_writer, key.c_str(), val) != 0) {
            throw std::runtime_error("Failed to add float metadata: " + key);
        }
    }

    void add_metadata(const std::string& key, bool val) {
        if (hk_writer_add_metadata_bool(m_writer, key.c_str(), val ? 1 : 0) != 0) {
            throw std::runtime_error("Failed to add bool metadata: " + key);
        }
    }

    void add_tensor(
        const std::string& name,
        StorageType storage_type,
        TileLayout tile_layout,
        SparsityType sparsity_type,
        std::span<const uint64_t> shape,
        std::span<const uint8_t> data,
        float sparsity_ratio = 0.0f
    ) {
        if (shape.size() > 8) {
            throw std::invalid_argument("HK supports at most 8 dimensions");
        }
        int res = hk_writer_add_tensor(
            m_writer,
            name.c_str(),
            static_cast<uint8_t>(storage_type),
            static_cast<uint8_t>(tile_layout),
            static_cast<uint8_t>(sparsity_type),
            static_cast<uint8_t>(shape.size()),
            shape.data(),
            data.data(),
            data.size(),
            sparsity_ratio
        );
        if (res != 0) {
            throw std::runtime_error("Failed to add tensor: " + name);
        }
    }

    void write_to_file(const std::string& path) {
        if (hk_writer_write_to_file(m_writer, path.c_str()) != 0) {
            throw std::runtime_error("Failed to write HK container to: " + path);
        }
    }

private:
    hk_writer_t* m_writer;
};

// SIMD Compute Operations
inline float dot(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector size mismatch");
    return hk_dot_product_f32(a.data(), b.data(), a.size());
}

inline void gemv(
    std::span<const float> W,
    std::span<const float> x,
    std::optional<std::span<const float>> bias,
    std::span<float> y,
    uint64_t M,
    uint64_t K
) {
    const float* b_ptr = bias ? bias->data() : nullptr;
    hk_gemv_f32(W.data(), x.data(), b_ptr, y.data(), M, K);
}

inline void gemm(
    std::span<const float> A,
    std::span<const float> B,
    std::span<float> C,
    uint64_t M,
    uint64_t K,
    uint64_t N
) {
    hk_gemm_f32(A.data(), B.data(), C.data(), M, K, N);
}

inline void fused_gemv_nf4(
    std::span<const uint8_t> packed_W,
    std::span<const float> scales,
    std::span<const float> x,
    std::optional<std::span<const float>> bias,
    std::span<float> y,
    uint64_t M,
    uint64_t K,
    uint32_t block_size = 32
) {
    const float* b_ptr = bias ? bias->data() : nullptr;
    hk_fused_gemv_nf4(packed_W.data(), scales.data(), x.data(), b_ptr, y.data(), M, K, block_size);
}

inline void fused_gemv_dq8(
    std::span<const int8_t> W_i8,
    std::span<const float> scales,
    std::span<const float> x,
    std::optional<std::span<const float>> bias,
    std::span<float> y,
    uint64_t M,
    uint64_t K,
    uint32_t block_size = 32
) {
    const float* b_ptr = bias ? bias->data() : nullptr;
    hk_fused_gemv_dq8(W_i8.data(), scales.data(), x.data(), b_ptr, y.data(), M, K, block_size);
}

inline int forward_swiglu(
    std::span<const float> x,
    std::span<const float> w_gate,
    std::optional<std::span<const float>> b_gate,
    std::span<const float> w_up,
    std::optional<std::span<const float>> b_up,
    std::span<const float> w_down,
    std::optional<std::span<const float>> b_down,
    std::span<float> intermediate_buf,
    std::span<float> out,
    uint64_t in_features,
    uint64_t inter_features,
    uint64_t out_features
) {
    return hk_forward_swiglu(
        x.data(),
        w_gate.data(),
        b_gate ? b_gate->data() : nullptr,
        w_up.data(),
        b_up ? b_up->data() : nullptr,
        w_down.data(),
        b_down ? b_down->data() : nullptr,
        intermediate_buf.data(),
        out.data(),
        in_features,
        inter_features,
        out_features
    );
}

inline void forward_rmsnorm(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out) {
    if (x.size() != weight.size() || x.size() != out.size()) {
        throw std::invalid_argument("Size mismatch in forward_rmsnorm");
    }
    hk_forward_rmsnorm(x.data(), weight.data(), eps, out.data(), x.size());
}

inline void forward_silu(std::span<const float> x, std::span<float> out) {
    if (x.size() != out.size()) {
        throw std::invalid_argument("Size mismatch in forward_silu");
    }
    hk_forward_silu(x.data(), out.data(), x.size());
}

// Net2Net Architecture Growth
inline int net2wider(
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
    float noise_std = 1e-4f,
    uint64_t seed = 42
) {
    return hk_net2wider(
        w_in_old, b_in_old, w_in_new, b_in_new,
        w_out_old, w_out_new,
        old_out, new_out, in_f, out_f,
        noise_std, seed
    );
}

inline void net2deeper(float* weights, float* bias, uint64_t dim) {
    hk_net2deeper(weights, bias, dim);
}

inline int net2wider_swiglu(
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
    bool zero_init = true,
    float noise_std = 1e-4f,
    uint64_t seed = 42
) {
    return hk_net2wider_swiglu(
        w_gate_old, w_gate_new, b_gate_old, b_gate_new,
        w_up_old, w_up_new, b_up_old, b_up_new,
        w_down_old, w_down_new, b_down_old, b_down_new,
        old_inter, new_inter, in_features, out_features,
        zero_init ? 1 : 0, noise_std, seed
    );
}

inline int expand_vocab(
    const float* embed_old,
    float* embed_new,
    const float* lm_head_old,
    float* lm_head_new,
    uint64_t old_vocab,
    uint64_t new_vocab,
    uint64_t hidden_dim,
    uint64_t seed = 42
) {
    return hk_expand_vocab(embed_old, embed_new, lm_head_old, lm_head_new, old_vocab, new_vocab, hidden_dim, seed);
}

inline void plasticity_mask_rows(std::span<float> grad, uint64_t cutoff_rows, uint64_t cols) {
    hk_plasticity_mask_rows(grad.data(), grad.size(), cutoff_rows, cols);
}

inline void plasticity_mask_cols(std::span<float> grad, uint64_t rows, uint64_t cutoff_cols, uint64_t cols) {
    hk_plasticity_mask_cols(grad.data(), grad.size(), rows, cutoff_cols, cols);
}

} // namespace hk

#endif // HK_HPP
