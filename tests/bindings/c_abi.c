// C ABI test and fixture generator for the language bindings.
//
//   c_abi write <out.hk>    writes the fixture every binding test reads
//   c_abi check <file.hk>   re-reads a file and asserts its contents
//   c_abi engine <tiny.hk>  runs the engine, tokenizer and sampler on a tiny model
//
// Build: cc -I include tests/bindings/c_abi.c -L zig-out/lib -lhk -o c_abi

#include "hk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static int write_fixture(const char* path) {
    hk_writer_t* w = hk_writer_create(4096);
    CHECK(w);
    CHECK(hk_writer_add_metadata_string(w, "general.name", "fixture") == 0);
    CHECK(hk_writer_add_metadata_int(w, "answer", 42) == 0);
    CHECK(hk_writer_add_metadata_float(w, "pi", 3.5) == 0);
    CHECK(hk_writer_add_metadata_bool(w, "flag", 1) == 0);

    float f[6] = {1, 2, 3, 4, 5, 6};
    uint64_t shape_f[2] = {2, 3};
    CHECK(hk_writer_add_tensor(w, "w.f32", HK_STORAGE_F32, HK_TILE_ROW_MAJOR, HK_SPARSITY_NONE, 2, shape_f, (const uint8_t*)f, sizeof f, 0.0f) == 0);

    float q[64];
    for (int i = 0; i < 64; i++) q[i] = (float)(i - 32) / 8.0f;
    uint8_t qb[2 * 34];
    CHECK(hk_quantize_tensor_q8_0(q, 64, qb) == 0);
    uint64_t shape_q[1] = {64};
    CHECK(hk_writer_add_tensor(w, "w.q8", HK_STORAGE_Q8_0, HK_TILE_ROW_MAJOR, HK_SPARSITY_NONE, 1, shape_q, qb, sizeof qb, 0.0f) == 0);
    CHECK(hk_writer_write_to_file(w, path) == 0);
    hk_writer_destroy(w);

    uint8_t zero[32] = {0};
    const char* payload = "adapter-bytes";
    CHECK(hk_appendix_append(path, HK_APPENDIX_LORA_ADAPTER, 1, "gen1", "w.f32", 1, zero, 1.5f, 0.5f, 0.25f, 0.0f, payload, strlen(payload)) == 0);
    return 0;
}

static int check_file(const char* path) {
    hk_reader_t* r = hk_open(path);
    CHECK(r);
    CHECK(hk_get_tensor_count(r) == 2);

    hk_tensor_info_t t;
    CHECK(hk_get_tensor_info(r, 0, &t) == 0);
    CHECK(strcmp(t.name, "w.f32") == 0);
    CHECK(t.storage_type == HK_STORAGE_F32 && t.ndim == 2 && t.shape[0] == 2 && t.shape[1] == 3);
    CHECK(t.data_offset % 4096 == 0);
    uint64_t sz = 0;
    const float* raw = hk_get_tensor_data(r, 0, &sz);
    CHECK(raw && sz == 24 && raw[0] == 1.0f && raw[5] == 6.0f);
    float out[6];
    CHECK(hk_dequantize_f32(r, 0, 1, out, 6) == 0);
    CHECK(out[2] == 3.0f);

    CHECK(hk_get_tensor_info(r, 1, &t) == 0);
    CHECK(strcmp(t.name, "w.q8") == 0 && t.storage_type == HK_STORAGE_Q8_0);
    float dq[64];
    CHECK(hk_dequantize_f32(r, 1, 0, dq, 64) == 0);
    for (int i = 0; i < 64; i++) CHECK(fabsf(dq[i] - (float)(i - 32) / 8.0f) < 0.02f);

    const char* s = hk_get_metadata_string(r, "general.name");
    CHECK(s && strcmp(s, "fixture") == 0);
    int64_t iv = 0; double fv = 0; int bv = 0;
    CHECK(hk_get_metadata_int(r, "answer", &iv) == 0 && iv == 42);
    CHECK(hk_get_metadata_float(r, "pi", &fv) == 0 && fv == 3.5);
    CHECK(hk_get_metadata_bool(r, "flag", &bv) == 0 && bv == 1);
    CHECK(hk_get_metadata_int(r, "missing", &iv) != 0);
    CHECK(hk_get_file_alignment(r) == 4096);
    CHECK(hk_is_universal_page_aligned(r) == 1);
    CHECK(hk_reader_is_sharded(r) == 0 && hk_reader_get_split_count(r) == 1);

    CHECK(hk_appendix_get_count(r) == 1);
    hk_appendix_entry_t e;
    CHECK(hk_appendix_get_entry(r, 0, &e) == 0);
    CHECK(e.entry_type == HK_APPENDIX_LORA_ADAPTER && e.generation == 1);
    CHECK(strcmp(e.name, "gen1") == 0 && strcmp(e.target, "w.f32") == 0);
    CHECK(e.data_size == 13 && memcmp(e.data, "adapter-bytes", 13) == 0);
    CHECK(e.metric_loss == 1.5f && e.metric_acc == 0.5f);
    hk_close(r);

    // 2:4 sparsity and NF4 helpers round trip.
    float d[8] = {0, 3, 0, 4, 5, 0, 6, 0};
    uint8_t packed[64];
    CHECK(hk_pack_2_4(d, 8, packed) == 0);
    float u[8];
    CHECK(hk_unpack_2_4(packed, 20, 8, u) == 0);
    for (int i = 0; i < 8; i++) CHECK(u[i] == d[i]);

    // Hardware profile and sampling.
    hk_hardware_caps_t caps;
    hk_detect_hardware(&caps);
    CHECK(caps.optimal_page_alignment >= 128);
    float logits[4] = {0.1f, 5.0f, 0.2f, 0.3f};
    CHECK(hk_sample_token(logits, 4, 0.0f, 0, 1.0f, 0.0f, 1.0f, NULL, 0, 0) == 1);
    return 0;
}

// Engine, tokenizer and sampler on a tiny model (made with hk-tiny-model, then hk convert-gguf).
static int check_engine(const char* path) {
    hk_engine_t* e = hk_engine_load_from_file(path);
    if (!e) {
        char msg[512];
        hk_engine_last_error(msg, sizeof msg);
        fprintf(stderr, "engine load failed: %s\n", msg);
        return 1;
    }
    uint32_t vocab = hk_engine_get_vocab_size(e);
    CHECK(vocab > 0 && hk_engine_get_context_size(e) > 4);
    float* logits = malloc(vocab * sizeof *logits);
    CHECK(logits);
    uint32_t prompt[3] = {1, 2, 3};
    CHECK(hk_engine_forward_tokens(e, prompt, 3, 0, logits) == 0);
    for (uint32_t i = 0; i < vocab; i++) CHECK(isfinite(logits[i]));
    // Feeding the same tokens one at a time must give the same last logits.
    hk_engine_reset_cache(e);
    float* again = malloc(vocab * sizeof *again);
    CHECK(again);
    for (uint32_t i = 0; i < 3; i++) CHECK(hk_engine_forward(e, prompt[i], i, again) == 0);
    for (uint32_t i = 0; i < vocab; i++) CHECK(fabsf(again[i] - logits[i]) < 1e-3f);
    CHECK(hk_engine_forward(e, 1, 1u << 30, again) == -2); // past the context window
    uint32_t t0 = hk_sample_token(logits, vocab, 0.0f, 0, 1.0f, 0.0f, 1.0f, NULL, 0, 0);
    CHECK(t0 < vocab);
    free(again);
    free(logits);
    hk_engine_free(e);

    hk_tokenizer_t* tok = hk_tokenizer_load_from_file(path);
    CHECK(tok && hk_tokenizer_get_vocab_size(tok) == vocab);
    uint32_t ids[64];
    uint32_t n = hk_tokenizer_encode(tok, "hello", 0, 0, ids, 64);
    CHECK(n > 0);
    uint8_t text[128];
    uint32_t len = hk_tokenizer_decode(tok, ids, n, 0, text, sizeof text);
    CHECK(len > 0);
    hk_tokenizer_free(tok);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && strcmp(argv[1], "engine") == 0) {
        int rc = check_engine(argv[2]);
        if (rc == 0) printf("c_abi engine ok\n");
        return rc;
    }
    if (argc != 3) { fprintf(stderr, "usage: c_abi write|check file.hk\n"); return 2; }
    int rc = strcmp(argv[1], "write") == 0 ? write_fixture(argv[2]) : check_file(argv[2]);
    if (rc == 0) printf("c_abi %s ok\n", argv[1]);
    return rc;
}
