// Exercises include/hk.hpp. Usage: cpp_test <fixture.hk> <scratch-dir>
#include "hk.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    auto model = hk::Model::open(argv[1]);
    CHECK(model->tensor_count() == 2);
    auto t = model->get_tensor("w.f32");
    CHECK(t && t->storage_type() == hk::StorageType::F32);
    CHECK((t->shape() == std::vector<uint64_t>{2, 3}));
    CHECK(t->element_count() == 6);
    auto raw = t->raw_data();
    CHECK(raw.size() == 24);
    auto f = t->dequantize();
    CHECK(f.size() == 6 && f[0] == 1.0f && f[5] == 6.0f);

    auto q = model->get_tensor(1);
    CHECK(q.name() == "w.q8" && q.storage_type() == hk::StorageType::Q8_0);
    auto qd = q.dequantize(false);
    CHECK(qd.size() == 64 && std::fabs(qd[0] + 4.0f) < 0.02f);

    CHECK(model->get_metadata_string("general.name") == "fixture");
    CHECK(model->get_metadata_int("answer") == 42);
    CHECK(model->get_metadata_float("pi") == 3.5);
    CHECK(model->get_metadata_bool("flag") == true);
    CHECK(!model->get_metadata_int("nope"));
    CHECK(model->appendix_count() == 1);
    auto e = model->get_appendix_entry(0);
    CHECK(e && std::string(e->name) == "gen1" && std::string(e->target) == "w.f32");

    // Write a file through the C++ wrapper and read it back.
    std::string out = (std::filesystem::path(argv[2]) / "cpp.hk").string();
    {
        hk::Writer w(128);
        w.add_metadata("general.name", "from-cpp");  // a literal must become a string, not a bool
        w.add_metadata("n", int64_t{7});
        float v[4] = {1, 2, 3, 4};
        uint64_t shape[2] = {2, 2};
        w.add_tensor("x", hk::StorageType::F32, hk::TileLayout::RowMajor, hk::SparsityType::None, shape,
                     std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(v), sizeof v));
        w.write_to_file(out);
    }
    auto back = hk::Model::open(out);
    CHECK(back->get_metadata_string("general.name") == "from-cpp");
    CHECK(back->get_metadata_int("n") == 7);
    CHECK(back->get_tensor("x")->dequantize()[3] == 4.0f);
    std::puts("cpp ok");
    return 0;
}
