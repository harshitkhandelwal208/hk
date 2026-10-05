package hk

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The fixture is written by tests/bindings/c_abi.c; set HK_FIXTURE to its path.
func fixture(t *testing.T) string {
	p := os.Getenv("HK_FIXTURE")
	if p == "" {
		t.Skip("HK_FIXTURE not set")
	}
	return p
}

func TestReadsFixture(t *testing.T) {
	m, err := Open(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.TensorCount() != 2 {
		t.Fatalf("tensor count %d", m.TensorCount())
	}
	w, err := m.GetTensor(0)
	if err != nil {
		t.Fatal(err)
	}
	if w.Name() != "w.f32" || w.StorageType() != StorageF32 || w.ElementCount() != 6 {
		t.Fatalf("unexpected tensor %s %v %d", w.Name(), w.StorageType(), w.ElementCount())
	}
	if s := w.Shape(); len(s) != 2 || s[0] != 2 || s[1] != 3 {
		t.Fatalf("shape %v", s)
	}
	vals, err := w.Dequantize(true)
	if err != nil || vals[5] != 6 {
		t.Fatalf("dequantize %v %v", vals, err)
	}
	q, _ := m.GetTensor(1)
	if q.StorageType() != StorageQ8_0 {
		t.Fatalf("storage %v", q.StorageType())
	}
	d, err := q.Dequantize(false)
	if err != nil || math.Abs(float64(d[0])+4) > 0.02 {
		t.Fatalf("q8 %v %v", d[:2], err)
	}

	if v, ok := m.GetMetadataString("general.name"); !ok || v != "fixture" {
		t.Fatalf("name %q %v", v, ok)
	}
	if v, ok := m.GetMetadataInt("answer"); !ok || v != 42 {
		t.Fatalf("int %v", v)
	}
	if v, ok := m.GetMetadataFloat("pi"); !ok || v != 3.5 {
		t.Fatalf("float %v", v)
	}
	if v, ok := m.GetMetadataBool("flag"); !ok || !v {
		t.Fatalf("bool %v", v)
	}
	if _, ok := m.GetMetadataInt("missing"); ok {
		t.Fatal("missing key found")
	}
	if m.FileAlignment() != 4096 || !m.IsUniversalPageAligned() || m.IsSharded() {
		t.Fatal("header flags")
	}
	if m.AppendixCount() != 1 {
		t.Fatal("appendix count")
	}
	e, err := m.GetAppendixEntry(0)
	if err != nil || e.Name != "gen1" || e.Target != "w.f32" || e.Generation != 1 || string(e.Data) != "adapter-bytes" {
		t.Fatalf("appendix %+v %v", e, err)
	}
}

func TestWriterAndPatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "w.hk")
	w, err := NewWriter(128)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddMetadataString("k", "v"); err != nil {
		t.Fatal(err)
	}
	data := []byte{0, 0, 0xc0, 0x3f, 0, 0, 0x20, 0x40} // 1.5, 2.5
	if err := w.AddTensor("t", StorageF32, TileRowMajor, SparsityNone, []uint64{2}, data, 0); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteToFile(path); err != nil {
		t.Fatal(err)
	}
	w.Close()
	if err := PatchMetadataInPlace(path, "k", "changed"); err != nil {
		t.Fatal(err)
	}
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if v, _ := m.GetMetadataString("k"); v != "changed" {
		t.Fatalf("patched value %q", v)
	}
	tn, _ := m.GetTensor(0)
	vals, _ := tn.Dequantize(false)
	if vals[0] != 1.5 || vals[1] != 2.5 {
		t.Fatalf("values %v", vals)
	}
}

func TestMathAndHardware(t *testing.T) {
	if DotF32([]float32{1, 2, 3}, []float32{4, 5, 6}) != 32 {
		t.Fatal("dot")
	}
	if DetectHardware().OptimalPageAlignment < 128 {
		t.Fatal("hardware")
	}
}
