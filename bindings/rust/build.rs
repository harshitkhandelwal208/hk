// Links the HK shared library (libhk.so / libhk.dylib / hk.dll).
//
// Build it with `zig build -Doptimize=ReleaseFast` in the repository root and point HK_LIB_DIR at
// the directory that holds it (zig-out/lib; on Windows also the import library hk.lib). Without
// HK_LIB_DIR the linker uses its default search path.

fn main() {
    println!("cargo:rerun-if-env-changed=HK_LIB_DIR");
    if let Ok(dir) = std::env::var("HK_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
        if !cfg!(windows) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
        }
    }
    println!("cargo:rustc-link-lib=dylib=hk");
}
