using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Hk
{
    public enum StorageType : byte
    {
        F32 = 0x00,
        F16 = 0x01,
        BF16 = 0x02,
        FP8_E4M3 = 0x03,
        FP8_E5M2 = 0x04,
        INT8 = 0x05,
        INT32 = 0x06,
        INT64 = 0x07,
        UINT8 = 0x08,
        BOOL = 0x09,
        INT16 = 0x0A,
        UINT16 = 0x0B,
        UINT32 = 0x0C,
        UINT64 = 0x0D,
        F64 = 0x0E,
        DQ4 = 0x10,
        NF4 = 0x10,
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
        NVFP4 = 0x63
    }

    public static class HkConstants
    {
        public const uint FlagIsSharded = 0x40;
        public const uint FlagRawWeightStorage = 1 << 7;
        public const uint FlagUniversalPageAligned = 1 << 8;

        public const ulong DefaultAlignmentBytes = 128;
        public const ulong UniversalPageAlignmentBytes = 4096;
        public const ulong AppleSiliconAlignmentBytes = 16384;
        public const ulong DirectDmaAlignmentBytes = 65536;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct HardwareCaps
    {
        public byte Vendor;
        public byte HasAvx2;
        public byte HasAvx512f;
        public byte HasAvx512vnni;
        public byte HasAvxVnni;
        public byte HasAmx;
        public byte HasArmNeon;
        public byte HasArmSve;
        public byte IsAppleSilicon;
        public byte HasRocmReady;
        public byte HasNpuReady;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 5)]
        public byte[] Reserved;
        public ulong OptimalPageAlignment;
        public ulong DmaHugepageAlignment;
    }

    public enum TileLayout : byte
    {
        RowMajor = 0x00,
        ColMajor = 0x01,
        Tile16x16 = 0x02,
        Tile16x8 = 0x03,
        Tile32x16 = 0x04,
        BlockSparse2_4 = 0x05,
        Tile32x32 = 0x06,
        Tile64x64 = 0x07
    }

    public enum SparsityType : byte
    {
        None = 0x00,
        Bitmask = 0x01,
        CSR = 0x02,
        Structured2_4 = 0x03,
        PhysicalPruned = 0x04,
        BSR = 0x05
    }

    public enum AppendixType : byte
    {
        LoRAAdapter = 0x01,
        DeltaPatch = 0x02,
        NewLayer = 0x03,
        CodeEval = 0x04,
        KVCacheSink = 0x05,
        TopologyHead = 0x06
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CTensorInfo
    {
        public IntPtr name;
        public byte storage_type;
        public byte tile_layout;
        public byte sparsity_type;
        public byte ndim;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public ulong[] shape;
        public ulong data_offset;
        public ulong data_size;
        public ulong residual_offset;
        public ulong residual_size;
        public ulong scale_offset;
        public ulong scale_size;
        public ushort block_size;
        public float sparsity_ratio;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CAppendixEntry
    {
        public byte entry_type;
        public byte flags;
        public uint generation;
        public ulong timestamp;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public byte[] parent_hash;
        public float metric_loss;
        public float metric_acc;
        public float metric_pass;
        public float metric_custom;
        public IntPtr name;
        public IntPtr target;
        public IntPtr data;
        public ulong data_size;
    }

    internal static class NativeMethods
    {
        private const string LibName = "hk";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_open([MarshalAs(UnmanagedType.LPUTF8Str)] string path);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_close(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong hk_get_tensor_count(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_get_tensor_info(IntPtr reader, ulong index, ref CTensorInfo out_info);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_data(IntPtr reader, ulong index, ref ulong out_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_residual(IntPtr reader, ulong index, ref ulong out_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_scales(IntPtr reader, ulong index, ref ulong out_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_dequantize_f32(IntPtr reader, ulong index, int with_residual, [Out] float[] out_buf, ulong count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_metadata_string(IntPtr reader, [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_get_metadata_int(IntPtr reader, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, ref long out_val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_get_metadata_float(IntPtr reader, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, ref double out_val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_get_metadata_bool(IntPtr reader, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, ref int out_val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_reader_is_sharded(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ushort hk_reader_get_split_index(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ushort hk_reader_get_split_count(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_metadata_patch_in_place([MarshalAs(UnmanagedType.LPUTF8Str)] string file_path, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, [MarshalAs(UnmanagedType.LPUTF8Str)] string val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_writer_set_sharding(IntPtr writer, ushort split_index, ushort split_count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong hk_appendix_get_count(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_appendix_get_entry(IntPtr reader, ulong index, ref CAppendixEntry out_entry);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_appendix_rollback([MarshalAs(UnmanagedType.LPUTF8Str)] string file_path, uint target_generation);

        // Compute & Kernels
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float hk_dot_product_f32([In] float[] a, [In] float[] b, ulong count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_gemv_f32([In] float[] W, [In] float[] x, [In] float[]? bias, [Out] float[] y, ulong M, ulong K);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_gemm_f32([In] float[] A, [In] float[] B, [Out] float[] C, ulong M, ulong K, ulong N);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_fused_gemv_nf4([In] byte[] packed_W, [In] float[] scales, [In] float[] x, [In] float[]? bias, [Out] float[] y, ulong M, ulong K, uint block_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_fused_gemv_dq8([In] sbyte[] W_i8, [In] float[] scales, [In] float[] x, [In] float[]? bias, [Out] float[] y, ulong M, ulong K, uint block_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_forward_swiglu([In] float[] x, [In] float[] w_gate, [In] float[]? b_gate, [In] float[] w_up, [In] float[]? b_up, [In] float[] w_down, [In] float[]? b_down, [Out] float[] intermediate_buf, [Out] float[] @out, ulong in_features, ulong inter_features, ulong out_features);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_forward_rmsnorm([In] float[] x, [In] float[] weight, float eps, [Out] float[] @out, ulong n);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_forward_silu([In] float[] x, [Out] float[] @out, ulong n);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_net2wider([In] float[] w_in_old, [In] float[]? b_in_old, [Out] float[] w_in_new, [Out] float[]? b_in_new, [In] float[] w_out_old, [Out] float[] w_out_new, ulong old_out, ulong new_out, ulong in_f, ulong out_f, float noise_std, ulong seed);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_net2deeper([In, Out] float[] weights, [In, Out] float[]? bias, ulong dim);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_net2wider_swiglu([In] float[] w_gate_old, [Out] float[] w_gate_new, [In] float[]? b_gate_old, [Out] float[]? b_gate_new, [In] float[] w_up_old, [Out] float[] w_up_new, [In] float[]? b_up_old, [Out] float[]? b_up_new, [In] float[] w_down_old, [Out] float[] w_down_new, [In] float[]? b_down_old, [Out] float[]? b_down_new, ulong old_inter, ulong new_inter, ulong in_features, ulong out_features, int zero_init, float noise_std, ulong seed);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_expand_vocab([In] float[] embed_old, [Out] float[] embed_new, [In] float[] lm_head_old, [Out] float[] lm_head_new, ulong old_vocab, ulong new_vocab, ulong hidden_dim, ulong seed);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_plasticity_mask_rows([In, Out] float[] grad, ulong total_elements, ulong cutoff_rows, ulong cols);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_plasticity_mask_cols([In, Out] float[] grad, ulong total_elements, ulong rows, ulong cutoff_cols, ulong cols);

        // Container Writer
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_writer_create(ulong alignment);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_writer_destroy(IntPtr writer);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_writer_add_metadata_string(IntPtr writer, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, [MarshalAs(UnmanagedType.LPUTF8Str)] string val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_writer_add_metadata_int(IntPtr writer, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, long val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_writer_add_metadata_float(IntPtr writer, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, double val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_writer_add_metadata_bool(IntPtr writer, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, int val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_writer_add_tensor(IntPtr writer, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, byte storage_type, byte tile_layout, byte sparsity_type, byte ndim, [In] ulong[] shape, [In] byte[] data, ulong data_len, float sparsity_ratio);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_writer_set_raw_storage(IntPtr writer, int enabled);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_writer_write_to_file(IntPtr writer, [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

        // Hardware Profiling & Zero-Copy Universal Alignment
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_detect_hardware(ref HardwareCaps out_caps);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong hk_get_optimal_alignment();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_is_raw_storage(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_is_universal_page_aligned(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint hk_get_file_alignment(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_raw_ptr(IntPtr reader, ulong index, ref ulong out_size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_raw_buffer(IntPtr reader, ref ulong out_size);

        // Raw Linear Algebra Kernels
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_gemv_bf16(IntPtr w_bf16, [In] float[] x, [In] float[]? bias, [Out] float[] y, ulong m, ulong k);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_gemv_f16(IntPtr w_f16, [In] float[] x, [In] float[]? bias, [Out] float[] y, ulong m, ulong k);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_gemv_int8(IntPtr w_i8, [In] float[] x, float scale_w, [In] float[]? bias, [Out] float[] y, ulong m, ulong k);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float hk_dot_bf16(IntPtr a, [In] float[] b, ulong len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float hk_dot_f16(IntPtr a, [In] float[] b, ulong len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_dot_int8(IntPtr a, IntPtr b, ulong len);
    }

    public sealed class HkTensor
    {
        private readonly IntPtr _reader;
        private readonly ulong _index;
        private readonly CTensorInfo _info;

        internal HkTensor(IntPtr reader, ulong index, CTensorInfo info)
        {
            _reader = reader;
            _index = index;
            _info = info;
            Name = Marshal.PtrToStringUTF8(info.name) ?? string.Empty;
            StorageType = (StorageType)info.storage_type;
            TileLayout = (TileLayout)info.tile_layout;
            SparsityType = (SparsityType)info.sparsity_type;
            Ndim = info.ndim;
            BlockSize = info.block_size;
            SparsityRatio = info.sparsity_ratio;
            Shape = new long[info.ndim];
            for (int i = 0; i < info.ndim; i++)
            {
                Shape[i] = (long)info.shape[i];
            }
        }

        public string Name { get; }
        public StorageType StorageType { get; }
        public TileLayout TileLayout { get; }
        public SparsityType SparsityType { get; }
        public byte Ndim { get; }
        public ushort BlockSize { get; }
        public float SparsityRatio { get; }
        public long[] Shape { get; }

        public long ElementCount
        {
            get
            {
                long cnt = 1;
                for (int i = 0; i < Ndim; i++) cnt *= Shape[i];
                return cnt;
            }
        }

        public float[] Dequantize(bool withResidual = true)
        {
            long count = ElementCount;
            float[] buffer = new float[count];
            int res = NativeMethods.hk_dequantize_f32(_reader, _index, withResidual ? 1 : 0, buffer, (ulong)count);
            if (res != 0)
            {
                throw new InvalidOperationException($"Failed to dequantize tensor {Name}");
            }
            return buffer;
        }

        public byte[] GetRawData()
        {
            ulong size = 0;
            IntPtr ptr = NativeMethods.hk_get_tensor_data(_reader, _index, ref size);
            if (ptr == IntPtr.Zero || size == 0) return Array.Empty<byte>();

            byte[] bytes = new byte[size];
            Marshal.Copy(ptr, bytes, 0, (int)size);
            return bytes;
        }

        public byte[] GetRawResidual()
        {
            ulong size = 0;
            IntPtr ptr = NativeMethods.hk_get_tensor_residual(_reader, _index, ref size);
            if (ptr == IntPtr.Zero || size == 0) return Array.Empty<byte>();

            byte[] bytes = new byte[size];
            Marshal.Copy(ptr, bytes, 0, (int)size);
            return bytes;
        }

        public byte[] GetRawScales()
        {
            ulong size = 0;
            IntPtr ptr = NativeMethods.hk_get_tensor_scales(_reader, _index, ref size);
            if (ptr == IntPtr.Zero || size == 0) return Array.Empty<byte>();

            byte[] bytes = new byte[size];
            Marshal.Copy(ptr, bytes, 0, (int)size);
            return bytes;
        }

        public bool IsRaw
        {
            get
            {
                return StorageType == StorageType.F32 ||
                       StorageType == StorageType.F16 ||
                       StorageType == StorageType.BF16 ||
                       StorageType == StorageType.FP8_E4M3 ||
                       StorageType == StorageType.FP8_E5M2 ||
                       StorageType == StorageType.INT8 ||
                       StorageType == StorageType.INT16 ||
                       StorageType == StorageType.INT32 ||
                       StorageType == StorageType.INT64 ||
                       StorageType == StorageType.UINT8 ||
                       StorageType == StorageType.UINT16 ||
                       StorageType == StorageType.UINT32 ||
                       StorageType == StorageType.UINT64 ||
                       StorageType == StorageType.BOOL ||
                       StorageType == StorageType.F64;
            }
        }

        public byte[] GetRawBytes()
        {
            ulong size = 0;
            IntPtr ptr = NativeMethods.hk_get_tensor_raw_ptr(_reader, _index, ref size);
            if (ptr == IntPtr.Zero || size == 0) return Array.Empty<byte>();

            byte[] bytes = new byte[size];
            Marshal.Copy(ptr, bytes, 0, (int)size);
            return bytes;
        }

        public IntPtr GetRawPointer(out ulong size)
        {
            size = 0;
            return NativeMethods.hk_get_tensor_raw_ptr(_reader, _index, ref size);
        }
    }

    public sealed class HkAppendixEntry
    {
        public AppendixType EntryType { get; }
        public byte Flags { get; }
        public uint Generation { get; }
        public ulong Timestamp { get; }
        public byte[] ParentHash { get; }
        public float MetricLoss { get; }
        public float MetricAcc { get; }
        public float MetricPass { get; }
        public float MetricCustom { get; }
        public string Name { get; }
        public string Target { get; }
        public ulong DataSize { get; }

        /// <summary>A copy of the record's payload.</summary>
        public byte[] Data { get; }

        internal HkAppendixEntry(CAppendixEntry entry)
        {
            EntryType = (AppendixType)entry.entry_type;
            Flags = entry.flags;
            Generation = entry.generation;
            Timestamp = entry.timestamp;
            ParentHash = (byte[])entry.parent_hash.Clone();
            MetricLoss = entry.metric_loss;
            MetricAcc = entry.metric_acc;
            MetricPass = entry.metric_pass;
            MetricCustom = entry.metric_custom;
            Name = Marshal.PtrToStringUTF8(entry.name) ?? string.Empty;
            Target = Marshal.PtrToStringUTF8(entry.target) ?? string.Empty;
            DataSize = entry.data_size;
            if (entry.data != IntPtr.Zero && entry.data_size > 0)
            {
                Data = new byte[entry.data_size];
                Marshal.Copy(entry.data, Data, 0, (int)entry.data_size);
            }
            else
            {
                Data = Array.Empty<byte>();
            }
        }
    }

    public sealed class HkModel : IDisposable
    {
        private IntPtr _reader;
        private bool _disposed;

        private HkModel(IntPtr reader)
        {
            _reader = reader;
        }

        public static HkModel Open(string path)
        {
            IntPtr r = NativeMethods.hk_open(path);
            if (r == IntPtr.Zero)
            {
                throw new System.IO.FileNotFoundException($"Could not open HK model file: {path}");
            }
            return new HkModel(r);
        }

        public ulong TensorCount
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_get_tensor_count(_reader);
            }
        }

        public HkTensor GetTensor(ulong index)
        {
            ThrowIfDisposed();
            var info = new CTensorInfo();
            int res = NativeMethods.hk_get_tensor_info(_reader, index, ref info);
            if (res != 0) throw new ArgumentOutOfRangeException(nameof(index));
            return new HkTensor(_reader, index, info);
        }

        public string? GetMetadataString(string key)
        {
            ThrowIfDisposed();
            IntPtr ptr = NativeMethods.hk_get_metadata_string(_reader, key);
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        public long? GetMetadataInt(string key)
        {
            ThrowIfDisposed();
            long val = 0;
            int res = NativeMethods.hk_get_metadata_int(_reader, key, ref val);
            return res == 0 ? (long?)val : null;
        }

        public double? GetMetadataFloat(string key)
        {
            ThrowIfDisposed();
            double val = 0.0;
            int res = NativeMethods.hk_get_metadata_float(_reader, key, ref val);
            return res == 0 ? (double?)val : null;
        }

        public bool? GetMetadataBool(string key)
        {
            ThrowIfDisposed();
            int val = 0;
            int res = NativeMethods.hk_get_metadata_bool(_reader, key, ref val);
            return res == 0 ? (bool?)(val != 0) : null;
        }

        public bool IsSharded
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_reader_is_sharded(_reader) != 0;
            }
        }

        public ushort SplitIndex
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_reader_get_split_index(_reader);
            }
        }

        public ushort SplitCount
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_reader_get_split_count(_reader);
            }
        }

        public static bool PatchMetadataInPlace(string path, string key, string val)
        {
            return NativeMethods.hk_metadata_patch_in_place(path, key, val) == 0;
        }

        public ulong AppendixCount
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_appendix_get_count(_reader);
            }
        }

        public HkAppendixEntry GetAppendixEntry(ulong index)
        {
            ThrowIfDisposed();
            var entry = new CAppendixEntry();
            int res = NativeMethods.hk_appendix_get_entry(_reader, index, ref entry);
            if (res != 0) throw new ArgumentOutOfRangeException(nameof(index));
            return new HkAppendixEntry(entry);
        }

        public static void Rollback(string path, uint targetGeneration)
        {
            int res = NativeMethods.hk_appendix_rollback(path, targetGeneration);
            if (res != 0)
            {
                throw new InvalidOperationException($"Failed to rollback {path} to generation {targetGeneration}");
            }
        }

        public bool IsRawStorage
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_is_raw_storage(_reader) != 0;
            }
        }

        public bool IsUniversalPageAligned
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_is_universal_page_aligned(_reader) != 0;
            }
        }

        public uint FileAlignment
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.hk_get_file_alignment(_reader);
            }
        }

        public bool IsTensorCoreAligned => (FileAlignment % 128) == 0;

        public static HardwareCaps DetectHardware()
        {
            var caps = new HardwareCaps();
            NativeMethods.hk_detect_hardware(ref caps);
            return caps;
        }

        public static ulong GetOptimalAlignment() => NativeMethods.hk_get_optimal_alignment();

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HkModel));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_reader != IntPtr.Zero)
                {
                    NativeMethods.hk_close(_reader);
                    _reader = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }

    public sealed class HkWriter : IDisposable
    {
        private IntPtr _writer;
        private bool _disposed;

        public HkWriter(ulong alignment = 64)
        {
            _writer = NativeMethods.hk_writer_create(alignment);
            if (_writer == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create HK writer");
            }
        }

        public void SetSharding(ushort splitIndex, ushort splitCount)
        {
            ThrowIfDisposed();
            NativeMethods.hk_writer_set_sharding(_writer, splitIndex, splitCount);
        }

        public void SetRawStorage(bool enabled)
        {
            ThrowIfDisposed();
            NativeMethods.hk_writer_set_raw_storage(_writer, enabled ? 1 : 0);
        }

        public void AddMetadataString(string key, string val)
        {
            ThrowIfDisposed();
            if (NativeMethods.hk_writer_add_metadata_string(_writer, key, val) != 0)
            {
                throw new InvalidOperationException($"Failed to add string metadata: {key}");
            }
        }

        public void AddMetadataInt(string key, long val)
        {
            ThrowIfDisposed();
            if (NativeMethods.hk_writer_add_metadata_int(_writer, key, val) != 0)
            {
                throw new InvalidOperationException($"Failed to add int metadata: {key}");
            }
        }

        public void AddMetadataFloat(string key, double val)
        {
            ThrowIfDisposed();
            if (NativeMethods.hk_writer_add_metadata_float(_writer, key, val) != 0)
            {
                throw new InvalidOperationException($"Failed to add float metadata: {key}");
            }
        }

        public void AddMetadataBool(string key, bool val)
        {
            ThrowIfDisposed();
            if (NativeMethods.hk_writer_add_metadata_bool(_writer, key, val ? 1 : 0) != 0)
            {
                throw new InvalidOperationException($"Failed to add bool metadata: {key}");
            }
        }

        public void AddTensor(string name, StorageType storage, TileLayout layout, SparsityType sparsity, ulong[] shape, byte[] data, float sparsityRatio = 0.0f)
        {
            ThrowIfDisposed();
            if (shape.Length > 8)
            {
                throw new ArgumentException("HK format supports at most 8 dimensions", nameof(shape));
            }
            int res = NativeMethods.hk_writer_add_tensor(_writer, name, (byte)storage, (byte)layout, (byte)sparsity, (byte)shape.Length, shape, data, (ulong)data.Length, sparsityRatio);
            if (res != 0)
            {
                throw new InvalidOperationException($"Failed to add tensor: {name}");
            }
        }

        public void WriteToFile(string path)
        {
            ThrowIfDisposed();
            int res = NativeMethods.hk_writer_write_to_file(_writer, path);
            if (res != 0)
            {
                throw new System.IO.IOException($"Failed to write HK container file: {path}");
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HkWriter));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_writer != IntPtr.Zero)
                {
                    NativeMethods.hk_writer_destroy(_writer);
                    _writer = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }
}

