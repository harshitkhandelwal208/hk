package hk

/*
#cgo CFLAGS: -I../../../include
#cgo LDFLAGS: -L${SRCDIR}/../../../zig-out/lib -lhk
#cgo linux LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../zig-out/lib
#cgo darwin LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../zig-out/lib
#include "hk.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type StorageType byte

const (
	StorageF32         StorageType = 0x00
	StorageF16         StorageType = 0x01
	StorageBF16        StorageType = 0x02
	StorageFP8E4M3     StorageType = 0x03
	StorageFP8E5M2     StorageType = 0x04
	StorageInt8        StorageType = 0x05
	StorageInt32       StorageType = 0x06
	StorageInt64       StorageType = 0x07
	StorageUint8       StorageType = 0x08
	StorageBool        StorageType = 0x09
	StorageInt16       StorageType = 0x0A
	StorageUint16      StorageType = 0x0B
	StorageUint32      StorageType = 0x0C
	StorageUint64      StorageType = 0x0D
	StorageF64         StorageType = 0x0E
	StorageDQ4         StorageType = 0x10
	StorageNF4         StorageType = 0x10
	StorageDQ8         StorageType = 0x11
	StorageDQ6         StorageType = 0x12
	StorageDQ12        StorageType = 0x13
	StorageDQT         StorageType = 0x14
	StorageQ4_0        StorageType = 0x15
	StorageQ8_0        StorageType = 0x16
	StorageQ4_1        StorageType = 0x17
	StorageQ5_0        StorageType = 0x18
	StorageQ5_1        StorageType = 0x19
	StorageQ8_1        StorageType = 0x1A
	StorageSparseF16   StorageType = 0x20
	StorageSparseDQ8   StorageType = 0x21
	StorageSparse24    StorageType = 0x22
	StorageSparseDQ424 StorageType = 0x23
	StorageNullRef     StorageType = 0x30
	StorageSharedRef   StorageType = 0x31
	StorageLoRARef     StorageType = 0x32
	StorageQ2_K        StorageType = 0x40
	StorageQ3_K        StorageType = 0x41
	StorageQ4_K        StorageType = 0x42
	StorageQ5_K        StorageType = 0x43
	StorageQ6_K        StorageType = 0x44
	StorageQ8_K        StorageType = 0x45
	StorageIQ1_S       StorageType = 0x50
	StorageIQ1_M       StorageType = 0x51
	StorageIQ2_XXS     StorageType = 0x52
	StorageIQ2_XS      StorageType = 0x53
	StorageIQ3_XXS     StorageType = 0x54
	StorageIQ4_NL      StorageType = 0x55
	StorageIQ4_XS      StorageType = 0x56
	StorageIQ2_S       StorageType = 0x57
	StorageIQ3_S       StorageType = 0x58
	StorageTQ1_0       StorageType = 0x60
	StorageTQ2_0       StorageType = 0x61
	StorageMXFP4       StorageType = 0x62
	StorageNVFP4       StorageType = 0x63
)

const (
	FlagIsSharded               = 0x40
	FlagRawWeightStorage        = 1 << 7
	FlagUniversalPageAligned    = 1 << 8
	DefaultAlignmentBytes       = 128
	UniversalPageAlignmentBytes = 4096
	AppleSiliconAlignmentBytes  = 16384
	DirectDMAAlignmentBytes     = 65536
)

type TileLayout byte

const (
	TileRowMajor      TileLayout = 0x00
	TileColMajor      TileLayout = 0x01
	Tile16x16         TileLayout = 0x02
	Tile16x8          TileLayout = 0x03
	Tile32x16         TileLayout = 0x04
	TileBlockSparse24 TileLayout = 0x05
	Tile32x32         TileLayout = 0x06
	Tile64x64         TileLayout = 0x07
)

type SparsityType byte

const (
	SparsityNone           SparsityType = 0x00
	SparsityBitmask        SparsityType = 0x01
	SparsityCSR            SparsityType = 0x02
	SparsityStructured24   SparsityType = 0x03
	SparsityPhysicalPruned SparsityType = 0x04
	SparsityBSR            SparsityType = 0x05
)

type AppendixType byte

const (
	AppendixLoRAAdapter  AppendixType = 0x01
	AppendixDeltaPatch   AppendixType = 0x02
	AppendixNewLayer     AppendixType = 0x03
	AppendixCodeEval     AppendixType = 0x04
	AppendixKVCacheSink  AppendixType = 0x05
	AppendixTopologyHead AppendixType = 0x06
)

type Tensor struct {
	model *Model
	index C.uint64_t
	info  C.hk_tensor_info_t
}

func (t *Tensor) Name() string {
	return C.GoString(t.info.name)
}

func (t *Tensor) StorageType() StorageType {
	return StorageType(t.info.storage_type)
}

func (t *Tensor) TileLayout() TileLayout {
	return TileLayout(t.info.tile_layout)
}

func (t *Tensor) SparsityType() SparsityType {
	return SparsityType(t.info.sparsity_type)
}

func (t *Tensor) BlockSize() uint16 {
	return uint16(t.info.block_size)
}

func (t *Tensor) SparsityRatio() float32 {
	return float32(t.info.sparsity_ratio)
}

func (t *Tensor) Shape() []uint64 {
	ndim := int(t.info.ndim)
	shape := make([]uint64, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = uint64(t.info.shape[i])
	}
	return shape
}

func (t *Tensor) ElementCount() uint64 {
	shape := t.Shape()
	count := uint64(1)
	for _, dim := range shape {
		count *= dim
	}
	return count
}

func (t *Tensor) Dequantize(withResidual bool) ([]float32, error) {
	count := t.ElementCount()
	buffer := make([]float32, count)

	withResInt := C.int(0)
	if withResidual {
		withResInt = C.int(1)
	}

	res := C.hk_dequantize_f32(
		t.model.handle,
		t.index,
		withResInt,
		(*C.float)(&buffer[0]),
		C.uint64_t(count),
	)
	if res != 0 {
		return nil, fmt.Errorf("dequantization failed for tensor %s", t.Name())
	}
	return buffer, nil
}

func (t *Tensor) RawData() []byte {
	var size C.uint64_t
	ptr := C.hk_get_tensor_data(t.model.handle, t.index, &size)
	if ptr == nil || size == 0 {
		return nil
	}
	return C.GoBytes(ptr, C.int(size))
}

func (t *Tensor) RawResidual() []byte {
	var size C.uint64_t
	ptr := C.hk_get_tensor_residual(t.model.handle, t.index, &size)
	if ptr == nil || size == 0 {
		return nil
	}
	return C.GoBytes(ptr, C.int(size))
}

func (t *Tensor) RawScales() []byte {
	var size C.uint64_t
	ptr := C.hk_get_tensor_scales(t.model.handle, t.index, &size)
	if ptr == nil || size == 0 {
		return nil
	}
	return C.GoBytes(ptr, C.int(size))
}

func (t *Tensor) RawBytes() []byte {
	var size C.uint64_t
	ptr := C.hk_get_tensor_raw_ptr(t.model.handle, t.index, &size)
	if ptr == nil || size == 0 {
		return nil
	}
	return C.GoBytes(ptr, C.int(size))
}

type AppendixEntry struct {
	EntryType    AppendixType
	Flags        uint8
	Generation   uint32
	Timestamp    uint64
	ParentHash   [32]byte
	MetricLoss   float32
	MetricAcc    float32
	MetricPass   float32
	MetricCustom float32
	Name         string
	Target       string
	DataSize     uint64
	// Data is a copy of the record's payload.
	Data []byte
}

type Model struct {
	handle *C.hk_reader_t
}

func Open(path string) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.hk_open(cPath)
	if handle == nil {
		return nil, fmt.Errorf("failed to open HK model file: %s", path)
	}

	m := &Model{handle: handle}
	runtime.SetFinalizer(m, (*Model).Close)
	return m, nil
}

func (m *Model) Close() {
	if m.handle != nil {
		C.hk_close(m.handle)
		m.handle = nil
	}
}

func (m *Model) TensorCount() int {
	return int(C.hk_get_tensor_count(m.handle))
}

func (m *Model) GetTensor(index int) (*Tensor, error) {
	var info C.hk_tensor_info_t
	res := C.hk_get_tensor_info(m.handle, C.uint64_t(index), &info)
	if res != 0 {
		return nil, errors.New("tensor index out of range")
	}
	return &Tensor{model: m, index: C.uint64_t(index), info: info}, nil
}

func (m *Model) GetMetadataString(key string) (string, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	cVal := C.hk_get_metadata_string(m.handle, cKey)
	if cVal == nil {
		return "", false
	}
	return C.GoString(cVal), true
}

func (m *Model) GetMetadataInt(key string) (int64, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var outVal C.int64_t
	res := C.hk_get_metadata_int(m.handle, cKey, &outVal)
	if res != 0 {
		return 0, false
	}
	return int64(outVal), true
}

func (m *Model) GetMetadataFloat(key string) (float64, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var outVal C.double
	res := C.hk_get_metadata_float(m.handle, cKey, &outVal)
	if res != 0 {
		return 0, false
	}
	return float64(outVal), true
}

func (m *Model) GetMetadataBool(key string) (bool, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var outVal C.int
	res := C.hk_get_metadata_bool(m.handle, cKey, &outVal)
	if res != 0 {
		return false, false
	}
	return outVal != 0, true
}

func (m *Model) IsSharded() bool {
	return C.hk_reader_is_sharded(m.handle) != 0
}

func (m *Model) SplitIndex() uint16 {
	return uint16(C.hk_reader_get_split_index(m.handle))
}

func (m *Model) SplitCount() uint16 {
	return uint16(C.hk_reader_get_split_count(m.handle))
}

func (m *Model) IsRawStorage() bool {
	return C.hk_is_raw_storage(m.handle) != 0
}

func (m *Model) IsUniversalPageAligned() bool {
	return C.hk_is_universal_page_aligned(m.handle) != 0
}

func (m *Model) FileAlignment() uint32 {
	return uint32(C.hk_get_file_alignment(m.handle))
}

func (m *Model) IsTensorCoreAligned() bool {
	return (m.FileAlignment() % 128) == 0
}

func PatchMetadataInPlace(path, key, val string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cVal := C.CString(val)
	defer C.free(unsafe.Pointer(cVal))

	if C.hk_metadata_patch_in_place(cPath, cKey, cVal) != 0 {
		return fmt.Errorf("in-place metadata patch failed for key '%s' in '%s'", key, path)
	}
	return nil
}

func (m *Model) AppendixCount() int {
	return int(C.hk_appendix_get_count(m.handle))
}

func (m *Model) GetAppendixEntry(index int) (*AppendixEntry, error) {
	var cEntry C.hk_appendix_entry_t
	res := C.hk_appendix_get_entry(m.handle, C.uint64_t(index), &cEntry)
	if res != 0 {
		return nil, errors.New("appendix entry not found")
	}

	entry := &AppendixEntry{
		EntryType:    AppendixType(cEntry.entry_type),
		Flags:        uint8(cEntry.flags),
		Generation:   uint32(cEntry.generation),
		Timestamp:    uint64(cEntry.timestamp),
		MetricLoss:   float32(cEntry.metric_loss),
		MetricAcc:    float32(cEntry.metric_acc),
		MetricPass:   float32(cEntry.metric_pass),
		MetricCustom: float32(cEntry.metric_custom),
		Name:         C.GoString(cEntry.name),
		Target:       C.GoString(cEntry.target),
		DataSize:     uint64(cEntry.data_size),
	}
	if cEntry.data != nil && cEntry.data_size > 0 {
		entry.Data = C.GoBytes(cEntry.data, C.int(cEntry.data_size))
	}
	for i := 0; i < 32; i++ {
		entry.ParentHash[i] = byte(cEntry.parent_hash[i])
	}
	return entry, nil
}

func (m *Model) Rollback(path string, targetGeneration uint32) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	res := C.hk_appendix_rollback(cPath, C.uint32_t(targetGeneration))
	if res != 0 {
		return fmt.Errorf("failed to rollback %s to generation %d", path, targetGeneration)
	}
	return nil
}

// Writer for creating .hk container files
type Writer struct {
	handle *C.hk_writer_t
}

func NewWriter(alignment uint64) (*Writer, error) {
	h := C.hk_writer_create(C.uint64_t(alignment))
	if h == nil {
		return nil, errors.New("failed to create hk_writer")
	}
	w := &Writer{handle: h}
	runtime.SetFinalizer(w, (*Writer).Close)
	return w, nil
}

func (w *Writer) Close() {
	if w.handle != nil {
		C.hk_writer_destroy(w.handle)
		w.handle = nil
	}
}

func (w *Writer) AddMetadataString(key, val string) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cVal := C.CString(val)
	defer C.free(unsafe.Pointer(cVal))

	if C.hk_writer_add_metadata_string(w.handle, cKey, cVal) != 0 {
		return fmt.Errorf("failed to add metadata string %s", key)
	}
	return nil
}

func (w *Writer) AddMetadataInt(key string, val int64) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	if C.hk_writer_add_metadata_int(w.handle, cKey, C.int64_t(val)) != 0 {
		return fmt.Errorf("failed to add metadata int %s", key)
	}
	return nil
}

func (w *Writer) AddMetadataFloat(key string, val float64) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	if C.hk_writer_add_metadata_float(w.handle, cKey, C.double(val)) != 0 {
		return fmt.Errorf("failed to add metadata float %s", key)
	}
	return nil
}

func (w *Writer) AddMetadataBool(key string, val bool) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	bVal := C.int(0)
	if val {
		bVal = C.int(1)
	}

	if C.hk_writer_add_metadata_bool(w.handle, cKey, bVal) != 0 {
		return fmt.Errorf("failed to add metadata bool %s", key)
	}
	return nil
}

func (w *Writer) AddTensor(
	name string,
	storage StorageType,
	tileLayout TileLayout,
	sparsity SparsityType,
	shape []uint64,
	data []byte,
	sparsityRatio float32,
) error {
	if len(shape) > 8 {
		return errors.New("cannot have more than 8 dimensions")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	cShape := make([]C.uint64_t, len(shape))
	for i, s := range shape {
		cShape[i] = C.uint64_t(s)
	}

	var shapePtr *C.uint64_t
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}

	var dataPtr *C.uint8_t
	if len(data) > 0 {
		dataPtr = (*C.uint8_t)(&data[0])
	}

	res := C.hk_writer_add_tensor(
		w.handle,
		cName,
		C.uint8_t(storage),
		C.uint8_t(tileLayout),
		C.uint8_t(sparsity),
		C.uint8_t(len(shape)),
		shapePtr,
		dataPtr,
		C.uint64_t(len(data)),
		C.float(sparsityRatio),
	)
	if res != 0 {
		return fmt.Errorf("failed to add tensor %s", name)
	}
	return nil
}

func (w *Writer) SetSharding(splitIndex, splitCount uint16) {
	C.hk_writer_set_sharding(w.handle, C.uint16_t(splitIndex), C.uint16_t(splitCount))
}

func (w *Writer) SetRawStorage(enabled bool) {
	b := C.int(0)
	if enabled {
		b = C.int(1)
	}
	C.hk_writer_set_raw_storage(w.handle, b)
}

func (w *Writer) WriteToFile(path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	if C.hk_writer_write_to_file(w.handle, cPath) != 0 {
		return fmt.Errorf("failed to write container to %s", path)
	}
	return nil
}

// SIMD Accelerated Compute Helpers
func DotF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("slice length mismatch")
	}
	return float32(C.hk_dot_product_f32((*C.float)(&a[0]), (*C.float)(&b[0]), C.uint64_t(len(a))))
}

func GemvF32(W, x []float32, bias []float32, y []float32, M, K int) {
	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(&bias[0])
	}
	C.hk_gemv_f32(
		(*C.float)(&W[0]),
		(*C.float)(&x[0]),
		biasPtr,
		(*C.float)(&y[0]),
		C.uint64_t(M),
		C.uint64_t(K),
	)
}

func GemmF32(A, B, C_out []float32, M, K, N int) {
	C.hk_gemm_f32(
		(*C.float)(&A[0]),
		(*C.float)(&B[0]),
		(*C.float)(&C_out[0]),
		C.uint64_t(M),
		C.uint64_t(K),
		C.uint64_t(N),
	)
}

func FusedGemvNF4(packedW []byte, scales []float32, x []float32, bias []float32, y []float32, M, K int, blockSize uint32) {
	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(&bias[0])
	}
	C.hk_fused_gemv_nf4(
		(*C.uint8_t)(&packedW[0]),
		(*C.float)(&scales[0]),
		(*C.float)(&x[0]),
		biasPtr,
		(*C.float)(&y[0]),
		C.uint64_t(M),
		C.uint64_t(K),
		C.uint32_t(blockSize),
	)
}

func FusedGemvDQ8(wI8 []int8, scales []float32, x []float32, bias []float32, y []float32, M, K int, blockSize uint32) {
	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(&bias[0])
	}
	C.hk_fused_gemv_dq8(
		(*C.int8_t)(&wI8[0]),
		(*C.float)(&scales[0]),
		(*C.float)(&x[0]),
		biasPtr,
		(*C.float)(&y[0]),
		C.uint64_t(M),
		C.uint64_t(K),
		C.uint32_t(blockSize),
	)
}

func ForwardSwiGLU(
	x []float32,
	wGate []float32,
	bGate []float32,
	wUp []float32,
	bUp []float32,
	wDown []float32,
	bDown []float32,
	intermediateBuf []float32,
	out []float32,
	inFeatures, interFeatures, outFeatures int,
) error {
	var bgPtr, buPtr, bdPtr *C.float
	if len(bGate) > 0 {
		bgPtr = (*C.float)(&bGate[0])
	}
	if len(bUp) > 0 {
		buPtr = (*C.float)(&bUp[0])
	}
	if len(bDown) > 0 {
		bdPtr = (*C.float)(&bDown[0])
	}

	res := C.hk_forward_swiglu(
		(*C.float)(&x[0]),
		(*C.float)(&wGate[0]),
		bgPtr,
		(*C.float)(&wUp[0]),
		buPtr,
		(*C.float)(&wDown[0]),
		bdPtr,
		(*C.float)(&intermediateBuf[0]),
		(*C.float)(&out[0]),
		C.uint64_t(inFeatures),
		C.uint64_t(interFeatures),
		C.uint64_t(outFeatures),
	)
	if res != 0 {
		return errors.New("SwiGLU execution failed")
	}
	return nil
}

func ForwardRMSNorm(x, weight []float32, eps float32, out []float32) {
	C.hk_forward_rmsnorm(
		(*C.float)(&x[0]),
		(*C.float)(&weight[0]),
		C.float(eps),
		(*C.float)(&out[0]),
		C.uint64_t(len(x)),
	)
}

func ForwardSiLU(x, out []float32) {
	C.hk_forward_silu(
		(*C.float)(&x[0]),
		(*C.float)(&out[0]),
		C.uint64_t(len(x)),
	)
}

func Net2Wider(
	wInOld, bInOld, wInNew, bInNew []float32,
	wOutOld, wOutNew []float32,
	oldOut, newOut, inF, outF int,
	noiseStd float32,
	seed uint64,
) error {
	var bioPtr, binPtr *C.float
	if len(bInOld) > 0 {
		bioPtr = (*C.float)(&bInOld[0])
	}
	if len(bInNew) > 0 {
		binPtr = (*C.float)(&bInNew[0])
	}

	res := C.hk_net2wider(
		(*C.float)(&wInOld[0]),
		bioPtr,
		(*C.float)(&wInNew[0]),
		binPtr,
		(*C.float)(&wOutOld[0]),
		(*C.float)(&wOutNew[0]),
		C.uint64_t(oldOut),
		C.uint64_t(newOut),
		C.uint64_t(inF),
		C.uint64_t(outF),
		C.float(noiseStd),
		C.uint64_t(seed),
	)
	if res != 0 {
		return errors.New("Net2Wider expansion failed")
	}
	return nil
}

func Net2Deeper(weights, bias []float32, dim int) {
	var bPtr *C.float
	if len(bias) > 0 {
		bPtr = (*C.float)(&bias[0])
	}
	C.hk_net2deeper((*C.float)(&weights[0]), bPtr, C.uint64_t(dim))
}

func Net2WiderSwiGLU(
	wGateOld, wGateNew, bGateOld, bGateNew []float32,
	wUpOld, wUpNew, bUpOld, bUpNew []float32,
	wDownOld, wDownNew, bDownOld, bDownNew []float32,
	oldInter, newInter, inFeatures, outFeatures int,
	zeroInit bool,
	noiseStd float32,
	seed uint64,
) error {
	var bgoPtr, bgnPtr, buoPtr, bunPtr, bdoPtr, bdnPtr *C.float
	if len(bGateOld) > 0 {
		bgoPtr = (*C.float)(&bGateOld[0])
	}
	if len(bGateNew) > 0 {
		bgnPtr = (*C.float)(&bGateNew[0])
	}
	if len(bUpOld) > 0 {
		buoPtr = (*C.float)(&bUpOld[0])
	}
	if len(bUpNew) > 0 {
		bunPtr = (*C.float)(&bUpNew[0])
	}
	if len(bDownOld) > 0 {
		bdoPtr = (*C.float)(&bDownOld[0])
	}
	if len(bDownNew) > 0 {
		bdnPtr = (*C.float)(&bDownNew[0])
	}

	zInit := C.int(0)
	if zeroInit {
		zInit = C.int(1)
	}

	res := C.hk_net2wider_swiglu(
		(*C.float)(&wGateOld[0]),
		(*C.float)(&wGateNew[0]),
		bgoPtr,
		bgnPtr,
		(*C.float)(&wUpOld[0]),
		(*C.float)(&wUpNew[0]),
		buoPtr,
		bunPtr,
		(*C.float)(&wDownOld[0]),
		(*C.float)(&wDownNew[0]),
		bdoPtr,
		bdnPtr,
		C.uint64_t(oldInter),
		C.uint64_t(newInter),
		C.uint64_t(inFeatures),
		C.uint64_t(outFeatures),
		zInit,
		C.float(noiseStd),
		C.uint64_t(seed),
	)
	if res != 0 {
		return errors.New("Net2Wider SwiGLU expansion failed")
	}
	return nil
}

func ExpandVocab(
	embedOld, embedNew, lmHeadOld, lmHeadNew []float32,
	oldVocab, newVocab, hiddenDim int,
	seed uint64,
) error {
	res := C.hk_expand_vocab(
		(*C.float)(&embedOld[0]),
		(*C.float)(&embedNew[0]),
		(*C.float)(&lmHeadOld[0]),
		(*C.float)(&lmHeadNew[0]),
		C.uint64_t(oldVocab),
		C.uint64_t(newVocab),
		C.uint64_t(hiddenDim),
		C.uint64_t(seed),
	)
	if res != 0 {
		return errors.New("Vocab expansion failed")
	}
	return nil
}

func PlasticityMaskRows(grad []float32, cutoffRows, cols int) {
	C.hk_plasticity_mask_rows((*C.float)(&grad[0]), C.uint64_t(len(grad)), C.uint64_t(cutoffRows), C.uint64_t(cols))
}

func PlasticityMaskCols(grad []float32, rows, cutoffCols, cols int) {
	C.hk_plasticity_mask_cols((*C.float)(&grad[0]), C.uint64_t(len(grad)), C.uint64_t(rows), C.uint64_t(cutoffCols), C.uint64_t(cols))
}

type HardwareCaps struct {
	Vendor               uint8
	HasAVX2              uint8
	HasAVX512F           uint8
	HasAVX512VNNI        uint8
	HasAVXVNNI           uint8
	HasAMX               uint8
	HasARMNeon           uint8
	HasARMSVE            uint8
	IsAppleSilicon       uint8
	HasROCmReady         uint8
	HasNPUReady          uint8
	Reserved             [5]uint8
	OptimalPageAlignment uint64
	DMAHugepageAlignment uint64
}

func DetectHardware() HardwareCaps {
	var cCaps C.hk_hardware_caps_t
	C.hk_detect_hardware(&cCaps)
	var caps HardwareCaps
	caps.Vendor = uint8(cCaps.vendor)
	caps.HasAVX2 = uint8(cCaps.has_avx2)
	caps.HasAVX512F = uint8(cCaps.has_avx512f)
	caps.HasAVX512VNNI = uint8(cCaps.has_avx512vnni)
	caps.HasAVXVNNI = uint8(cCaps.has_avx_vnni)
	caps.HasAMX = uint8(cCaps.has_amx)
	caps.HasARMNeon = uint8(cCaps.has_arm_neon)
	caps.HasARMSVE = uint8(cCaps.has_arm_sve)
	caps.IsAppleSilicon = uint8(cCaps.is_apple_silicon)
	caps.HasROCmReady = uint8(cCaps.has_rocm_ready)
	caps.HasNPUReady = uint8(cCaps.has_npu_ready)
	caps.OptimalPageAlignment = uint64(cCaps.optimal_page_alignment)
	caps.DMAHugepageAlignment = uint64(cCaps.dma_hugepage_alignment)
	return caps
}

func GetOptimalAlignment() uint64 {
	return uint64(C.hk_get_optimal_alignment())
}

func GemvBF16(w []uint16, x []float32, bias []float32, y []float32, m, k int) {
	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(&bias[0])
	}
	C.hk_gemv_bf16(
		(*C.uint16_t)(&w[0]),
		(*C.float)(&x[0]),
		biasPtr,
		(*C.float)(&y[0]),
		C.size_t(m),
		C.size_t(k),
	)
}

func GemvF16(w []uint16, x []float32, bias []float32, y []float32, m, k int) {
	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(&bias[0])
	}
	C.hk_gemv_f16(
		unsafe.Pointer(&w[0]),
		(*C.float)(&x[0]),
		biasPtr,
		(*C.float)(&y[0]),
		C.size_t(m),
		C.size_t(k),
	)
}

func GemvInt8(w []int8, x []float32, scaleW float32, bias []float32, y []float32, m, k int) {
	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(&bias[0])
	}
	C.hk_gemv_int8(
		(*C.int8_t)(&w[0]),
		(*C.float)(&x[0]),
		C.float(scaleW),
		biasPtr,
		(*C.float)(&y[0]),
		C.size_t(m),
		C.size_t(k),
	)
}

func DotBF16(a []uint16, b []float32) float32 {
	if len(a) != len(b) {
		panic("length mismatch")
	}
	return float32(C.hk_dot_bf16((*C.uint16_t)(&a[0]), (*C.float)(&b[0]), C.size_t(len(a))))
}

func DotF16(a []uint16, b []float32) float32 {
	if len(a) != len(b) {
		panic("length mismatch")
	}
	return float32(C.hk_dot_f16(unsafe.Pointer(&a[0]), (*C.float)(&b[0]), C.size_t(len(a))))
}

func DotInt8(a, b []int8) int32 {
	if len(a) != len(b) {
		panic("length mismatch")
	}
	return int32(C.hk_dot_int8((*C.int8_t)(&a[0]), (*C.int8_t)(&b[0]), C.size_t(len(a))))
}
