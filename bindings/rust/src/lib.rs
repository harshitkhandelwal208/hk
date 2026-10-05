//! Safe idiomatic Rust bindings for the HK neural tensor format and SIMD compute engine.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum StorageType {
    F32 = 0x00,
    F16 = 0x01,
    BF16 = 0x02,
    FP8E4M3 = 0x03,
    FP8E5M2 = 0x04,
    Int8 = 0x05,
    Int32 = 0x06,
    Int64 = 0x07,
    UInt8 = 0x08,
    Bool = 0x09,
    Int16 = 0x0A,
    UInt16 = 0x0B,
    UInt32 = 0x0C,
    UInt64 = 0x0D,
    F64 = 0x0E,
    DQ4 = 0x10,
    DQ8 = 0x11,
    DQ6 = 0x12,
    DQ12 = 0x13,
    DQT = 0x14,
    Q4_0 = 0x15,
    Q8_0 = 0x16,
    Q4_1 = 0x17,
    Q5_0 = 0x18,
    Q5_1 = 0x19,
    Q8_1 = 0x1A,
    SparseF16 = 0x20,
    SparseDQ8 = 0x21,
    Sparse24 = 0x22,
    SparseDQ4_2_4 = 0x23,
    NullRef = 0x30,
    SharedRef = 0x31,
    LoRARef = 0x32,
    Q2_K = 0x40,
    Q3_K = 0x41,
    Q4_K = 0x42,
    Q5_K = 0x43,
    Q6_K = 0x44,
    Q8_K = 0x45,
    IQ1_S = 0x50,
    IQ1_M = 0x51,
    IQ2_XXS = 0x52,
    IQ2_XS = 0x53,
    IQ3_XXS = 0x54,
    IQ4_NL = 0x55,
    IQ4_XS = 0x56,
    IQ2_S = 0x57,
    IQ3_S = 0x58,
    TQ1_0 = 0x60,
    TQ2_0 = 0x61,
    MXFP4 = 0x62,
    NVFP4 = 0x63,
}

pub const NF4: StorageType = StorageType::DQ4;
pub const FLAG_IS_SHARDED: u32 = 0x40;
pub const FLAG_RAW_WEIGHT_STORAGE: u32 = 1 << 7;
pub const FLAG_UNIVERSAL_PAGE_ALIGNED: u32 = 1 << 8;

pub const DEFAULT_ALIGNMENT_BYTES: u64 = 128;
pub const UNIVERSAL_PAGE_ALIGNMENT_BYTES: u64 = 4096;
pub const APPLE_SILICON_ALIGNMENT_BYTES: u64 = 16384;
pub const DIRECT_DMA_ALIGNMENT_BYTES: u64 = 65536;

impl StorageType {
    pub fn is_raw(&self) -> bool {
        matches!(
            self,
            StorageType::F32
                | StorageType::F16
                | StorageType::BF16
                | StorageType::FP8E4M3
                | StorageType::FP8E5M2
                | StorageType::Int8
                | StorageType::Int16
                | StorageType::Int32
                | StorageType::Int64
                | StorageType::UInt8
                | StorageType::UInt16
                | StorageType::UInt32
                | StorageType::UInt64
                | StorageType::Bool
                | StorageType::F64
        )
    }

    pub fn element_size(&self) -> usize {
        match self {
            StorageType::Bool | StorageType::UInt8 | StorageType::Int8 | StorageType::FP8E4M3 | StorageType::FP8E5M2 => 1,
            StorageType::F16 | StorageType::BF16 | StorageType::Int16 | StorageType::UInt16 => 2,
            StorageType::F32 | StorageType::Int32 | StorageType::UInt32 => 4,
            StorageType::Int64 | StorageType::UInt64 | StorageType::F64 => 8,
            _ => 0,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLayout {
    RowMajor = 0x00,
    ColMajor = 0x01,
    Tile16x16 = 0x02,
    Tile16x8 = 0x03,
    Tile32x16 = 0x04,
    BlockSparse2_4 = 0x05,
    Tile32x32 = 0x06,
    Tile64x64 = 0x07,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsityType {
    None = 0x00,
    Bitmask = 0x01,
    Csr = 0x02,
    Structured2_4 = 0x03,
    PhysicalPruned = 0x04,
    Bsr = 0x05,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendixType {
    LoRAAdapter = 0x01,
    DeltaPatch = 0x02,
    NewLayer = 0x03,
    CodeEval = 0x04,
    KVCacheSink = 0x05,
    TopologyHead = 0x06,
}

#[repr(C)]
pub struct CTensorInfo {
    pub name: *const c_char,
    pub storage_type: u8,
    pub tile_layout: u8,
    pub sparsity_type: u8,
    pub ndim: u8,
    pub shape: [u64; 8],
    pub data_offset: u64,
    pub data_size: u64,
    pub residual_offset: u64,
    pub residual_size: u64,
    pub scale_offset: u64,
    pub scale_size: u64,
    pub block_size: u16,
    pub sparsity_ratio: f32,
}

#[repr(C)]
pub struct CAppendixEntry {
    pub entry_type: u8,
    pub flags: u8,
    pub generation: u32,
    pub timestamp: u64,
    pub parent_hash: [u8; 32],
    pub metric_loss: f32,
    pub metric_acc: f32,
    pub metric_pass: f32,
    pub metric_custom: f32,
    pub name: *const c_char,
    pub target: *const c_char,
    pub data: *const c_void,
    pub data_size: u64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct HardwareCaps {
    pub vendor: u8,
    pub has_avx2: u8,
    pub has_avx512f: u8,
    pub has_avx512vnni: u8,
    pub has_avx_vnni: u8,
    pub has_amx: u8,
    pub has_arm_neon: u8,
    pub has_arm_sve: u8,
    pub is_apple_silicon: u8,
    pub has_rocm_ready: u8,
    pub has_npu_ready: u8,
    pub reserved: [u8; 5],
    pub optimal_page_alignment: u64,
    pub dma_hugepage_alignment: u64,
}

pub enum HkReaderOpaque {}
pub enum HkWriterOpaque {}

extern "C" {
    fn hk_open(path: *const c_char) -> *mut HkReaderOpaque;
    fn hk_close(reader: *mut HkReaderOpaque);
    fn hk_get_tensor_count(reader: *const HkReaderOpaque) -> u64;
    fn hk_get_tensor_info(reader: *const HkReaderOpaque, index: u64, out_info: *mut CTensorInfo) -> c_int;
    fn hk_get_tensor_data(reader: *const HkReaderOpaque, index: u64, out_size: *mut u64) -> *const c_void;
    fn hk_get_tensor_residual(reader: *const HkReaderOpaque, index: u64, out_size: *mut u64) -> *const c_void;
    fn hk_get_tensor_scales(reader: *const HkReaderOpaque, index: u64, out_size: *mut u64) -> *const c_void;
    fn hk_dequantize_f32(
        reader: *const HkReaderOpaque,
        index: u64,
        with_residual: c_int,
        out_buf: *mut f32,
        count: u64,
    ) -> c_int;
    fn hk_get_metadata_string(reader: *const HkReaderOpaque, key: *const c_char) -> *const c_char;
    fn hk_get_metadata_int(reader: *const HkReaderOpaque, key: *const c_char, out_val: *mut i64) -> c_int;
    fn hk_get_metadata_float(reader: *const HkReaderOpaque, key: *const c_char, out_val: *mut f64) -> c_int;
    fn hk_get_metadata_bool(reader: *const HkReaderOpaque, key: *const c_char, out_val: *mut c_int) -> c_int;

    fn hk_appendix_get_count(reader: *const HkReaderOpaque) -> u64;
    fn hk_appendix_get_entry(reader: *const HkReaderOpaque, index: u64, out_entry: *mut CAppendixEntry) -> c_int;
    fn hk_appendix_rollback(file_path: *const c_char, target_generation: u32) -> c_int;

    fn hk_reader_is_sharded(reader: *const HkReaderOpaque) -> c_int;
    fn hk_reader_get_split_index(reader: *const HkReaderOpaque) -> u16;
    fn hk_reader_get_split_count(reader: *const HkReaderOpaque) -> u16;
    fn hk_metadata_patch_in_place(file_path: *const c_char, key: *const c_char, val: *const c_char) -> c_int;

    // SIMD Compute
    fn hk_dot_product_f32(a: *const f32, b: *const f32, count: u64) -> f32;
    fn hk_gemv_f32(w: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: u64, k: u64);
    fn hk_gemm_f32(a: *const f32, b: *const f32, c: *mut f32, m: u64, k: u64, n: u64);
    fn hk_fused_gemv_nf4(
        packed_w: *const u8,
        scales: *const f32,
        x: *const f32,
        bias: *const f32,
        y: *mut f32,
        m: u64,
        k: u64,
        block_size: u32,
    );
    fn hk_fused_gemv_dq8(
        w_i8: *const i8,
        scales: *const f32,
        x: *const f32,
        bias: *const f32,
        y: *mut f32,
        m: u64,
        k: u64,
        block_size: u32,
    );
    fn hk_forward_swiglu(
        x: *const f32,
        w_gate: *const f32,
        b_gate: *const f32,
        w_up: *const f32,
        b_up: *const f32,
        w_down: *const f32,
        b_down: *const f32,
        intermediate_buf: *mut f32,
        out: *mut f32,
        in_features: u64,
        inter_features: u64,
        out_features: u64,
    ) -> c_int;
    fn hk_forward_rmsnorm(x: *const f32, weight: *const f32, eps: f32, out: *mut f32, n: u64);
    fn hk_forward_silu(x: *const f32, out: *mut f32, n: u64);

    // Growth
    fn hk_net2wider(
        w_in_old: *const f32,
        b_in_old: *const f32,
        w_in_new: *mut f32,
        b_in_new: *mut f32,
        w_out_old: *const f32,
        w_out_new: *mut f32,
        old_out: u64,
        new_out: u64,
        in_f: u64,
        out_f: u64,
        noise_std: f32,
        seed: u64,
    ) -> c_int;
    fn hk_net2deeper(weights: *mut f32, bias: *mut f32, dim: u64);
    fn hk_net2wider_swiglu(
        w_gate_old: *const f32,
        w_gate_new: *mut f32,
        b_gate_old: *const f32,
        b_gate_new: *mut f32,
        w_up_old: *const f32,
        w_up_new: *mut f32,
        b_up_old: *const f32,
        b_up_new: *mut f32,
        w_down_old: *const f32,
        w_down_new: *mut f32,
        b_down_old: *const f32,
        b_down_new: *mut f32,
        old_inter: u64,
        new_inter: u64,
        in_features: u64,
        out_features: u64,
        zero_init: c_int,
        noise_std: f32,
        seed: u64,
    ) -> c_int;
    fn hk_expand_vocab(
        embed_old: *const f32,
        embed_new: *mut f32,
        lm_head_old: *const f32,
        lm_head_new: *mut f32,
        old_vocab: u64,
        new_vocab: u64,
        hidden_dim: u64,
        seed: u64,
    ) -> c_int;
    fn hk_plasticity_mask_rows(grad: *mut f32, total_elements: u64, cutoff_rows: u64, cols: u64);
    fn hk_plasticity_mask_cols(grad: *mut f32, total_elements: u64, rows: u64, cutoff_cols: u64, cols: u64);

    // Writer
    fn hk_writer_create(alignment: u64) -> *mut HkWriterOpaque;
    fn hk_writer_destroy(writer: *mut HkWriterOpaque);
    fn hk_writer_set_sharding(writer: *mut HkWriterOpaque, split_index: u16, split_count: u16);
    fn hk_writer_add_metadata_string(writer: *mut HkWriterOpaque, key: *const c_char, val: *const c_char) -> c_int;
    fn hk_writer_add_metadata_int(writer: *mut HkWriterOpaque, key: *const c_char, val: i64) -> c_int;
    fn hk_writer_add_metadata_float(writer: *mut HkWriterOpaque, key: *const c_char, val: f64) -> c_int;
    fn hk_writer_add_metadata_bool(writer: *mut HkWriterOpaque, key: *const c_char, val: c_int) -> c_int;
    fn hk_writer_add_tensor(
        writer: *mut HkWriterOpaque,
        name: *const c_char,
        storage_type: u8,
        tile_layout: u8,
        sparsity_type: u8,
        ndim: u8,
        shape: *const u64,
        data: *const u8,
        data_len: u64,
        sparsity_ratio: f32,
    ) -> c_int;
    fn hk_writer_set_raw_storage(writer: *mut HkWriterOpaque, enabled: c_int);
    fn hk_writer_write_to_file(writer: *mut HkWriterOpaque, path: *const c_char) -> c_int;

    // Hardware Profiling & Zero-Copy Universal Alignment
    fn hk_detect_hardware(out_caps: *mut HardwareCaps);
    fn hk_get_optimal_alignment() -> usize;
    fn hk_is_raw_storage(reader: *const HkReaderOpaque) -> c_int;
    fn hk_is_universal_page_aligned(reader: *const HkReaderOpaque) -> c_int;
    fn hk_get_file_alignment(reader: *const HkReaderOpaque) -> u32;
    fn hk_get_tensor_raw_ptr(reader: *const HkReaderOpaque, index: u64, out_size: *mut u64) -> *const c_void;
    #[allow(dead_code)]
    fn hk_get_raw_buffer(reader: *const HkReaderOpaque, out_size: *mut u64) -> *const c_void;

    // Raw Weights Linear Algebra
    fn hk_gemv_bf16(w_bf16: *const u16, x: *const f32, bias: *const f32, y: *mut f32, m: usize, k: usize);
    fn hk_gemv_f16(w_f16: *const c_void, x: *const f32, bias: *const f32, y: *mut f32, m: usize, k: usize);
    fn hk_gemv_int8(w_i8: *const i8, x: *const f32, scale_w: f32, bias: *const f32, y: *mut f32, m: usize, k: usize);
    fn hk_dot_bf16(a: *const u16, b: *const f32, len: usize) -> f32;
    fn hk_dot_f16(a: *const c_void, b: *const f32, len: usize) -> f32;
    fn hk_dot_int8(a: *const i8, b: *const i8, len: usize) -> i32;
}

pub struct HkTensor<'a> {
    reader: &'a HkModel,
    index: u64,
    info: CTensorInfo,
}

impl<'a> HkTensor<'a> {
    pub fn name(&self) -> &str {
        if self.info.name.is_null() {
            ""
        } else {
            unsafe { CStr::from_ptr(self.info.name).to_str().unwrap_or("") }
        }
    }

    pub fn storage_type(&self) -> StorageType {
        match self.info.storage_type {
            0x00 => StorageType::F32,
            0x01 => StorageType::F16,
            0x02 => StorageType::BF16,
            0x03 => StorageType::FP8E4M3,
            0x04 => StorageType::FP8E5M2,
            0x05 => StorageType::Int8,
            0x06 => StorageType::Int32,
            0x07 => StorageType::Int64,
            0x08 => StorageType::UInt8,
            0x09 => StorageType::Bool,
            0x0A => StorageType::Int16,
            0x0B => StorageType::UInt16,
            0x0C => StorageType::UInt32,
            0x0D => StorageType::UInt64,
            0x0E => StorageType::F64,
            0x10 => StorageType::DQ4,
            0x11 => StorageType::DQ8,
            0x12 => StorageType::DQ6,
            0x13 => StorageType::DQ12,
            0x14 => StorageType::DQT,
            0x15 => StorageType::Q4_0,
            0x16 => StorageType::Q8_0,
            0x17 => StorageType::Q4_1,
            0x18 => StorageType::Q5_0,
            0x19 => StorageType::Q5_1,
            0x1A => StorageType::Q8_1,
            0x20 => StorageType::SparseF16,
            0x21 => StorageType::SparseDQ8,
            0x22 => StorageType::Sparse24,
            0x23 => StorageType::SparseDQ4_2_4,
            0x30 => StorageType::NullRef,
            0x31 => StorageType::SharedRef,
            0x32 => StorageType::LoRARef,
            0x40 => StorageType::Q2_K,
            0x41 => StorageType::Q3_K,
            0x42 => StorageType::Q4_K,
            0x43 => StorageType::Q5_K,
            0x44 => StorageType::Q6_K,
            0x45 => StorageType::Q8_K,
            0x50 => StorageType::IQ1_S,
            0x51 => StorageType::IQ1_M,
            0x52 => StorageType::IQ2_XXS,
            0x53 => StorageType::IQ2_XS,
            0x54 => StorageType::IQ3_XXS,
            0x55 => StorageType::IQ4_NL,
            0x56 => StorageType::IQ4_XS,
            0x57 => StorageType::IQ2_S,
            0x58 => StorageType::IQ3_S,
            0x60 => StorageType::TQ1_0,
            0x61 => StorageType::TQ2_0,
            0x62 => StorageType::MXFP4,
            0x63 => StorageType::NVFP4,
            _ => StorageType::F32,
        }
    }

    pub fn tile_layout(&self) -> TileLayout {
        match self.info.tile_layout {
            0x00 => TileLayout::RowMajor,
            0x01 => TileLayout::ColMajor,
            0x02 => TileLayout::Tile16x16,
            0x03 => TileLayout::Tile16x8,
            0x04 => TileLayout::Tile32x16,
            0x05 => TileLayout::BlockSparse2_4,
            0x06 => TileLayout::Tile32x32,
            0x07 => TileLayout::Tile64x64,
            _ => TileLayout::RowMajor,
        }
    }

    pub fn sparsity_type(&self) -> SparsityType {
        match self.info.sparsity_type {
            0x00 => SparsityType::None,
            0x01 => SparsityType::Bitmask,
            0x02 => SparsityType::Csr,
            0x03 => SparsityType::Structured2_4,
            0x04 => SparsityType::PhysicalPruned,
            0x05 => SparsityType::Bsr,
            _ => SparsityType::None,
        }
    }

    pub fn shape(&self) -> &[u64] {
        &self.info.shape[..self.info.ndim as usize]
    }

    pub fn element_count(&self) -> usize {
        self.shape().iter().product::<u64>() as usize
    }

    pub fn block_size(&self) -> u16 {
        self.info.block_size
    }

    pub fn sparsity_ratio(&self) -> f32 {
        self.info.sparsity_ratio
    }

    pub fn dequantize(&self, with_residual: bool) -> Result<Vec<f32>, String> {
        let count = self.element_count();
        let mut buffer = Vec::with_capacity(count);
        unsafe {
            let res = hk_dequantize_f32(
                self.reader.raw,
                self.index,
                if with_residual { 1 } else { 0 },
                buffer.as_mut_ptr(),
                count as u64,
            );
            if res != 0 {
                return Err(format!("Dequantization failed for tensor {}", self.name()));
            }
            buffer.set_len(count);
        }
        Ok(buffer)
    }

    pub fn raw_data(&self) -> &[u8] {
        let mut size = 0u64;
        unsafe {
            let ptr = hk_get_tensor_data(self.reader.raw, self.index, &mut size);
            if ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr as *const u8, size as usize)
            }
        }
    }

    pub fn raw_residual(&self) -> &[u8] {
        let mut size = 0u64;
        unsafe {
            let ptr = hk_get_tensor_residual(self.reader.raw, self.index, &mut size);
            if ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr as *const u8, size as usize)
            }
        }
    }

    pub fn raw_scales(&self) -> &[u8] {
        let mut size = 0u64;
        unsafe {
            let ptr = hk_get_tensor_scales(self.reader.raw, self.index, &mut size);
            if ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr as *const u8, size as usize)
            }
        }
    }

    pub fn is_raw(&self) -> bool {
        self.storage_type().is_raw()
    }

    pub fn raw_bytes(&self) -> &[u8] {
        let mut size = 0u64;
        unsafe {
            let ptr = hk_get_tensor_raw_ptr(self.reader.raw, self.index, &mut size);
            if ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr as *const u8, size as usize)
            }
        }
    }

    pub fn as_raw_f32(&self) -> Option<&[f32]> {
        if self.storage_type() == StorageType::F32 {
            let bytes = self.raw_bytes();
            let count = bytes.len() / std::mem::size_of::<f32>();
            Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) })
        } else {
            None
        }
    }

    pub fn as_raw_i8(&self) -> Option<&[i8]> {
        if self.storage_type() == StorageType::Int8 {
            let bytes = self.raw_bytes();
            Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) })
        } else {
            None
        }
    }

    pub fn as_raw_u16(&self) -> Option<&[u16]> {
        if matches!(self.storage_type(), StorageType::BF16 | StorageType::F16 | StorageType::UInt16 | StorageType::Int16) {
            let bytes = self.raw_bytes();
            let count = bytes.len() / 2;
            Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, count) })
        } else {
            None
        }
    }
}

pub struct AppendixEntry {
    pub entry_type: AppendixType,
    pub flags: u8,
    pub generation: u32,
    pub timestamp: u64,
    pub parent_hash: [u8; 32],
    pub metric_loss: f32,
    pub metric_acc: f32,
    pub metric_pass: f32,
    pub metric_custom: f32,
    pub name: String,
    pub target: String,
    pub data_size: u64,
    /// A copy of the record's payload.
    pub data: Vec<u8>,
}

pub struct HkModel {
    raw: *mut HkReaderOpaque,
}

impl HkModel {
    pub fn open(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let raw = unsafe { hk_open(c_path.as_ptr()) };
        if raw.is_null() {
            Err(format!("Failed to open HK model file at {}", path))
        } else {
            Ok(Self { raw })
        }
    }

    pub fn tensor_count(&self) -> usize {
        unsafe { hk_get_tensor_count(self.raw) as usize }
    }

    pub fn get_tensor(&self, index: usize) -> Option<HkTensor<'_>> {
        let mut info: CTensorInfo = unsafe { std::mem::zeroed() };
        let res = unsafe { hk_get_tensor_info(self.raw, index as u64, &mut info) };
        if res == 0 {
            Some(HkTensor {
                reader: self,
                index: index as u64,
                info,
            })
        } else {
            None
        }
    }

    pub fn get_metadata_string(&self, key: &str) -> Option<String> {
        let c_key = CString::new(key).ok()?;
        unsafe {
            let ptr = hk_get_metadata_string(self.raw, c_key.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
            }
        }
    }

    pub fn get_metadata_int(&self, key: &str) -> Option<i64> {
        let c_key = CString::new(key).ok()?;
        let mut val = 0i64;
        let res = unsafe { hk_get_metadata_int(self.raw, c_key.as_ptr(), &mut val) };
        if res == 0 {
            Some(val)
        } else {
            None
        }
    }

    pub fn get_metadata_float(&self, key: &str) -> Option<f64> {
        let c_key = CString::new(key).ok()?;
        let mut val = 0.0f64;
        let res = unsafe { hk_get_metadata_float(self.raw, c_key.as_ptr(), &mut val) };
        if res == 0 {
            Some(val)
        } else {
            None
        }
    }

    pub fn get_metadata_bool(&self, key: &str) -> Option<bool> {
        let c_key = CString::new(key).ok()?;
        let mut val: c_int = 0;
        let res = unsafe { hk_get_metadata_bool(self.raw, c_key.as_ptr(), &mut val) };
        if res == 0 {
            Some(val != 0)
        } else {
            None
        }
    }

    pub fn appendix_count(&self) -> u64 {
        unsafe { hk_appendix_get_count(self.raw) }
    }

    pub fn get_appendix_entry(&self, index: u64) -> Option<AppendixEntry> {
        let mut c_entry: CAppendixEntry = unsafe { std::mem::zeroed() };
        let res = unsafe { hk_appendix_get_entry(self.raw, index, &mut c_entry) };
        if res != 0 {
            return None;
        }
        let name = if c_entry.name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(c_entry.name).to_string_lossy().into_owned() }
        };
        let target = if c_entry.target.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(c_entry.target).to_string_lossy().into_owned() }
        };
        let entry_type = match c_entry.entry_type {
            0x01 => AppendixType::LoRAAdapter,
            0x02 => AppendixType::DeltaPatch,
            0x03 => AppendixType::NewLayer,
            0x04 => AppendixType::CodeEval,
            0x05 => AppendixType::KVCacheSink,
            0x06 => AppendixType::TopologyHead,
            _ => AppendixType::LoRAAdapter,
        };
        Some(AppendixEntry {
            entry_type,
            flags: c_entry.flags,
            generation: c_entry.generation,
            timestamp: c_entry.timestamp,
            parent_hash: c_entry.parent_hash,
            metric_loss: c_entry.metric_loss,
            metric_acc: c_entry.metric_acc,
            metric_pass: c_entry.metric_pass,
            metric_custom: c_entry.metric_custom,
            name,
            target,
            data_size: c_entry.data_size,
            data: if c_entry.data.is_null() || c_entry.data_size == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(c_entry.data as *const u8, c_entry.data_size as usize).to_vec() }
            },
        })
    }

    pub fn rollback(path: &str, target_generation: u32) -> Result<(), String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let res = unsafe { hk_appendix_rollback(c_path.as_ptr(), target_generation) };
        if res == 0 {
            Ok(())
        } else {
            Err(format!("Rollback to generation {} failed for {}", target_generation, path))
        }
    }

    pub fn is_sharded(&self) -> bool {
        unsafe { hk_reader_is_sharded(self.raw) != 0 }
    }

    pub fn split_index(&self) -> u16 {
        unsafe { hk_reader_get_split_index(self.raw) }
    }

    pub fn split_count(&self) -> u16 {
        unsafe { hk_reader_get_split_count(self.raw) }
    }

    pub fn patch_metadata_in_place(path: &str, key: &str, val: &str) -> Result<(), String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let c_key = CString::new(key).map_err(|e| e.to_string())?;
        let c_val = CString::new(val).map_err(|e| e.to_string())?;
        let res = unsafe { hk_metadata_patch_in_place(c_path.as_ptr(), c_key.as_ptr(), c_val.as_ptr()) };
        if res == 0 {
            Ok(())
        } else {
            Err(format!("In-place metadata patch failed for key '{}' in '{}'", key, path))
        }
    }

    pub fn is_raw_storage(&self) -> bool {
        unsafe { hk_is_raw_storage(self.raw) != 0 }
    }

    pub fn is_universal_page_aligned(&self) -> bool {
        unsafe { hk_is_universal_page_aligned(self.raw) != 0 }
    }

    pub fn file_alignment(&self) -> u32 {
        unsafe { hk_get_file_alignment(self.raw) }
    }

    pub fn is_tensor_core_aligned(&self) -> bool {
        (self.file_alignment() % 128) == 0
    }
}

impl Drop for HkModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { hk_close(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

// HkWriter for building .hk files
pub struct HkWriter {
    raw: *mut HkWriterOpaque,
}

impl HkWriter {
    pub fn new(alignment: u64) -> Result<Self, String> {
        let raw = unsafe { hk_writer_create(alignment) };
        if raw.is_null() {
            Err("Failed to create HkWriter".into())
        } else {
            Ok(Self { raw })
        }
    }

    pub fn set_sharding(&mut self, split_index: u16, split_count: u16) {
        unsafe { hk_writer_set_sharding(self.raw, split_index, split_count) }
    }

    pub fn set_raw_storage(&mut self, enabled: bool) {
        unsafe { hk_writer_set_raw_storage(self.raw, if enabled { 1 } else { 0 }); }
    }

    pub fn add_metadata_string(&mut self, key: &str, val: &str) -> Result<(), String> {
        let c_key = CString::new(key).map_err(|e| e.to_string())?;
        let c_val = CString::new(val).map_err(|e| e.to_string())?;
        let res = unsafe { hk_writer_add_metadata_string(self.raw, c_key.as_ptr(), c_val.as_ptr()) };
        if res == 0 { Ok(()) } else { Err(format!("Failed to add metadata string {}", key)) }
    }

    pub fn add_metadata_int(&mut self, key: &str, val: i64) -> Result<(), String> {
        let c_key = CString::new(key).map_err(|e| e.to_string())?;
        let res = unsafe { hk_writer_add_metadata_int(self.raw, c_key.as_ptr(), val) };
        if res == 0 { Ok(()) } else { Err(format!("Failed to add metadata int {}", key)) }
    }

    pub fn add_metadata_float(&mut self, key: &str, val: f64) -> Result<(), String> {
        let c_key = CString::new(key).map_err(|e| e.to_string())?;
        let res = unsafe { hk_writer_add_metadata_float(self.raw, c_key.as_ptr(), val) };
        if res == 0 { Ok(()) } else { Err(format!("Failed to add metadata float {}", key)) }
    }

    pub fn add_metadata_bool(&mut self, key: &str, val: bool) -> Result<(), String> {
        let c_key = CString::new(key).map_err(|e| e.to_string())?;
        let res = unsafe { hk_writer_add_metadata_bool(self.raw, c_key.as_ptr(), if val { 1 } else { 0 }) };
        if res == 0 { Ok(()) } else { Err(format!("Failed to add metadata bool {}", key)) }
    }

    pub fn add_tensor(
        &mut self,
        name: &str,
        storage_type: StorageType,
        tile_layout: TileLayout,
        sparsity_type: SparsityType,
        shape: &[u64],
        data: &[u8],
        sparsity_ratio: f32,
    ) -> Result<(), String> {
        if shape.len() > 8 {
            return Err("Shape cannot have more than 8 dimensions".into());
        }
        let c_name = CString::new(name).map_err(|e| e.to_string())?;
        let res = unsafe {
            hk_writer_add_tensor(
                self.raw,
                c_name.as_ptr(),
                storage_type as u8,
                tile_layout as u8,
                sparsity_type as u8,
                shape.len() as u8,
                shape.as_ptr(),
                data.as_ptr(),
                data.len() as u64,
                sparsity_ratio,
            )
        };
        if res == 0 { Ok(()) } else { Err(format!("Failed to add tensor {}", name)) }
    }

    pub fn write_to_file(&mut self, path: &str) -> Result<(), String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let res = unsafe { hk_writer_write_to_file(self.raw, c_path.as_ptr()) };
        if res == 0 { Ok(()) } else { Err(format!("Failed to write container to {}", path)) }
    }
}

impl Drop for HkWriter {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { hk_writer_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

// SIMD acceleration functions
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { hk_dot_product_f32(a.as_ptr(), b.as_ptr(), a.len() as u64) }
}

pub fn gemv_f32(w: &[f32], x: &[f32], bias: Option<&[f32]>, y: &mut [f32], m: usize, k: usize) {
    let bias_ptr = bias.map_or(ptr::null(), |b| b.as_ptr());
    unsafe { hk_gemv_f32(w.as_ptr(), x.as_ptr(), bias_ptr, y.as_mut_ptr(), m as u64, k as u64) }
}

pub fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe { hk_gemm_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as u64, k as u64, n as u64) }
}

pub fn fused_gemv_nf4(
    packed_w: &[u8],
    scales: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    block_size: u32,
) {
    let bias_ptr = bias.map_or(ptr::null(), |b| b.as_ptr());
    unsafe {
        hk_fused_gemv_nf4(
            packed_w.as_ptr(),
            scales.as_ptr(),
            x.as_ptr(),
            bias_ptr,
            y.as_mut_ptr(),
            m as u64,
            k as u64,
            block_size,
        )
    }
}

pub fn fused_gemv_dq8(
    w_i8: &[i8],
    scales: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    block_size: u32,
) {
    let bias_ptr = bias.map_or(ptr::null(), |b| b.as_ptr());
    unsafe {
        hk_fused_gemv_dq8(
            w_i8.as_ptr(),
            scales.as_ptr(),
            x.as_ptr(),
            bias_ptr,
            y.as_mut_ptr(),
            m as u64,
            k as u64,
            block_size,
        )
    }
}

pub fn forward_swiglu(
    x: &[f32],
    w_gate: &[f32],
    b_gate: Option<&[f32]>,
    w_up: &[f32],
    b_up: Option<&[f32]>,
    w_down: &[f32],
    b_down: Option<&[f32]>,
    intermediate_buf: &mut [f32],
    out: &mut [f32],
    in_features: usize,
    inter_features: usize,
    out_features: usize,
) -> Result<(), String> {
    let res = unsafe {
        hk_forward_swiglu(
            x.as_ptr(),
            w_gate.as_ptr(),
            b_gate.map_or(ptr::null(), |b| b.as_ptr()),
            w_up.as_ptr(),
            b_up.map_or(ptr::null(), |b| b.as_ptr()),
            w_down.as_ptr(),
            b_down.map_or(ptr::null(), |b| b.as_ptr()),
            intermediate_buf.as_mut_ptr(),
            out.as_mut_ptr(),
            in_features as u64,
            inter_features as u64,
            out_features as u64,
        )
    };
    if res == 0 { Ok(()) } else { Err("SwiGLU forward execution failed".into()) }
}

pub fn forward_rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    unsafe { hk_forward_rmsnorm(x.as_ptr(), weight.as_ptr(), eps, out.as_mut_ptr(), x.len() as u64) }
}

pub fn forward_silu(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    unsafe { hk_forward_silu(x.as_ptr(), out.as_mut_ptr(), x.len() as u64) }
}

pub fn net2wider(
    w_in_old: &[f32],
    b_in_old: &[f32],
    w_in_new: &mut [f32],
    b_in_new: &mut [f32],
    w_out_old: &[f32],
    w_out_new: &mut [f32],
    old_out: usize,
    new_out: usize,
    in_f: usize,
    out_f: usize,
    noise_std: f32,
    seed: u64,
) -> Result<(), String> {
    let res = unsafe {
        hk_net2wider(
            w_in_old.as_ptr(),
            b_in_old.as_ptr(),
            w_in_new.as_mut_ptr(),
            b_in_new.as_mut_ptr(),
            w_out_old.as_ptr(),
            w_out_new.as_mut_ptr(),
            old_out as u64,
            new_out as u64,
            in_f as u64,
            out_f as u64,
            noise_std,
            seed,
        )
    };
    if res == 0 { Ok(()) } else { Err("Net2Wider expansion failed".into()) }
}

pub fn net2deeper(weights: &mut [f32], bias: &mut [f32], dim: usize) {
    unsafe { hk_net2deeper(weights.as_mut_ptr(), bias.as_mut_ptr(), dim as u64) }
}

pub fn net2wider_swiglu(
    w_gate_old: &[f32],
    w_gate_new: &mut [f32],
    b_gate_old: &[f32],
    b_gate_new: &mut [f32],
    w_up_old: &[f32],
    w_up_new: &mut [f32],
    b_up_old: &[f32],
    b_up_new: &mut [f32],
    w_down_old: &[f32],
    w_down_new: &mut [f32],
    b_down_old: &[f32],
    b_down_new: &mut [f32],
    old_inter: usize,
    new_inter: usize,
    in_features: usize,
    out_features: usize,
    zero_init: bool,
    noise_std: f32,
    seed: u64,
) -> Result<(), String> {
    let res = unsafe {
        hk_net2wider_swiglu(
            w_gate_old.as_ptr(),
            w_gate_new.as_mut_ptr(),
            b_gate_old.as_ptr(),
            b_gate_new.as_mut_ptr(),
            w_up_old.as_ptr(),
            w_up_new.as_mut_ptr(),
            b_up_old.as_ptr(),
            b_up_new.as_mut_ptr(),
            w_down_old.as_ptr(),
            w_down_new.as_mut_ptr(),
            b_down_old.as_ptr(),
            b_down_new.as_mut_ptr(),
            old_inter as u64,
            new_inter as u64,
            in_features as u64,
            out_features as u64,
            if zero_init { 1 } else { 0 },
            noise_std,
            seed,
        )
    };
    if res == 0 { Ok(()) } else { Err("Net2Wider SwiGLU expansion failed".into()) }
}

pub fn expand_vocab(
    embed_old: &[f32],
    embed_new: &mut [f32],
    lm_head_old: &[f32],
    lm_head_new: &mut [f32],
    old_vocab: usize,
    new_vocab: usize,
    hidden_dim: usize,
    seed: u64,
) -> Result<(), String> {
    let res = unsafe {
        hk_expand_vocab(
            embed_old.as_ptr(),
            embed_new.as_mut_ptr(),
            lm_head_old.as_ptr(),
            lm_head_new.as_mut_ptr(),
            old_vocab as u64,
            new_vocab as u64,
            hidden_dim as u64,
            seed,
        )
    };
    if res == 0 { Ok(()) } else { Err("Vocab expansion failed".into()) }
}

pub fn plasticity_mask_rows(grad: &mut [f32], cutoff_rows: usize, cols: usize) {
    unsafe { hk_plasticity_mask_rows(grad.as_mut_ptr(), grad.len() as u64, cutoff_rows as u64, cols as u64) }
}

pub fn plasticity_mask_cols(grad: &mut [f32], rows: usize, cutoff_cols: usize, cols: usize) {
    unsafe { hk_plasticity_mask_cols(grad.as_mut_ptr(), grad.len() as u64, rows as u64, cutoff_cols as u64, cols as u64) }
}

// Hardware Profiling
pub fn detect_hardware() -> HardwareCaps {
    let mut caps: HardwareCaps = unsafe { std::mem::zeroed() };
    unsafe { hk_detect_hardware(&mut caps) };
    caps
}

pub fn get_optimal_alignment() -> usize {
    unsafe { hk_get_optimal_alignment() }
}

// Raw Weights Zero-Copy Linear Algebra
pub fn gemv_bf16(w_bf16: &[u16], x: &[f32], bias: Option<&[f32]>, y: &mut [f32], m: usize, k: usize) {
    let bias_ptr = bias.map_or(ptr::null(), |b| b.as_ptr());
    unsafe { hk_gemv_bf16(w_bf16.as_ptr(), x.as_ptr(), bias_ptr, y.as_mut_ptr(), m, k) }
}

pub fn gemv_f16(w_f16: &[u16], x: &[f32], bias: Option<&[f32]>, y: &mut [f32], m: usize, k: usize) {
    let bias_ptr = bias.map_or(ptr::null(), |b| b.as_ptr());
    unsafe { hk_gemv_f16(w_f16.as_ptr() as *const c_void, x.as_ptr(), bias_ptr, y.as_mut_ptr(), m, k) }
}

pub fn gemv_int8(w_i8: &[i8], x: &[f32], scale_w: f32, bias: Option<&[f32]>, y: &mut [f32], m: usize, k: usize) {
    let bias_ptr = bias.map_or(ptr::null(), |b| b.as_ptr());
    unsafe { hk_gemv_int8(w_i8.as_ptr(), x.as_ptr(), scale_w, bias_ptr, y.as_mut_ptr(), m, k) }
}

pub fn dot_bf16(a: &[u16], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { hk_dot_bf16(a.as_ptr(), b.as_ptr(), a.len()) }
}

pub fn dot_f16(a: &[u16], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { hk_dot_f16(a.as_ptr() as *const c_void, b.as_ptr(), a.len()) }
}

pub fn dot_int8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    unsafe { hk_dot_int8(a.as_ptr(), b.as_ptr(), a.len()) }
}


