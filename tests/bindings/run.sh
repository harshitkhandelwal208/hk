#!/usr/bin/env bash
# Builds libhk, writes the shared fixture through the C ABI, and runs every language binding's
# test against it. Languages whose toolchain is missing are skipped, unless HK_BINDINGS_STRICT=1.
#
#   tests/bindings/run.sh            (from the repository root)
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
LIB=$ROOT/zig-out/lib
export LD_LIBRARY_PATH=$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=$LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
export HK_LIB_DIR=$LIB
failed=0
ran=0

step() { printf '\n== %s\n' "$1"; }
run() {  # run <name> <command...>
  local name=$1; shift
  ran=$((ran + 1))
  if "$@"; then echo "[pass] $name"; else echo "[FAIL] $name"; failed=$((failed + 1)); fi
}
need() {  # need <tool>: returns 1 (and counts a failure in strict mode) when missing
  command -v "$1" >/dev/null 2>&1 && return 0
  if [ "${HK_BINDINGS_STRICT:-0}" = 1 ]; then echo "[FAIL] $1 not installed"; failed=$((failed + 1)); else echo "[skip] $1 not installed"; fi
  return 1
}

JDK=${JAVA_HOME:-}
if [ -z "$JDK" ] && command -v javac >/dev/null 2>&1; then
  JDK=$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")
fi
step "build libhk"
if [ -n "$JDK" ] && [ -f "$JDK/include/jni.h" ]; then
  zig build install jni -Doptimize=ReleaseFast -Djdk="$JDK" || exit 1
else
  zig build install -Doptimize=ReleaseFast || exit 1
fi

step "fixture"
CC=${CC:-cc}; CXX=${CXX:-c++}
$CC -std=c11 -Wall -Wextra -I include tests/bindings/c_abi.c -L "$LIB" -lhk -lm -o "$WORK/c_abi" || exit 1
"$WORK/c_abi" write "$WORK/fixture.hk" || exit 1
export HK_FIXTURE=$WORK/fixture.hk

step "C"
run "c abi" "$WORK/c_abi" check "$HK_FIXTURE"
"$ROOT/zig-out/bin/hk-tiny-model" "$WORK/tiny.gguf" && "$ROOT/zig-out/bin/hk" convert-gguf "$WORK/tiny.gguf" "$WORK/tiny.hk" >/dev/null 2>&1
run "c engine, tokenizer, sampler" "$WORK/c_abi" engine "$WORK/tiny.hk"

step "C++"
if need "$CXX"; then
  $CXX -std=c++20 -Wall -Wextra -I include tests/bindings/cpp_test.cpp -L "$LIB" -lhk -o "$WORK/cpp_test" \
    && run "c++" "$WORK/cpp_test" "$HK_FIXTURE" "$WORK" || { echo "[FAIL] c++ build"; failed=$((failed + 1)); }
fi

step "Rust"
if need cargo; then run "rust" bash -c "cd bindings/rust && cargo test --quiet"; fi

step "Go"
if need go; then run "go" bash -c "cd bindings/go && go vet ./... && go test ./..."; fi

step "C#"
if need dotnet; then run "c#" bash -c "cd bindings/csharp/tests && DOTNET_CLI_TELEMETRY_OPTOUT=1 dotnet run"; fi

step "Java"
if need javac && [ -f "$LIB/libhkjni.so" -o -f "$LIB/libhkjni.dylib" ]; then
  mkdir -p "$WORK/jout"
  javac -d "$WORK/jout" bindings/java/com/hk/HkModel.java bindings/java/test/HkModelTest.java \
    && run "java" java -Djava.library.path="$LIB" -cp "$WORK/jout" HkModelTest
else
  echo "[skip] java (no JDK with jni.h, so libhkjni was not built)"
fi

step "TypeScript"
if need node && need npm; then
  run "typescript" bash -c "cd bindings/js && npm install --no-audit --no-fund --silent && npx tsc && HK_CLI='$ROOT/zig-out/bin/hk' node test/test.mjs"
fi

printf '\n%d suites run, %d failed\n' "$ran" "$failed"
[ "$failed" -eq 0 ]
