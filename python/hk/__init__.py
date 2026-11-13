"""
hk: a from-scratch LLM runtime and model format.

`import hk` has no dependencies and imports nothing heavy. Everything below loads on first use (PEP 562), so the
command line, the native engine, tokenizers and the downloader work on a machine with neither NumPy nor PyTorch.
The PyTorch and NumPy based layers (`hk.torch`, `hk.numpy`, `HKForCausalLM`, `HKTrainer`, ...) need those packages
and say so when you first touch them: `pip install 'hknt[torch]'`.
"""

import importlib
from typing import Any

__version__ = "2.0.0"

# Public name -> (submodule, attribute). Nothing here is imported until the name is used.
_EXPORTS = {
    "HKConfig": ("config", "HKConfig"),
    "AutoConfig": ("config", "AutoConfig"),
    "HKPreTrainedModel": ("modeling", "HKPreTrainedModel"),
    "HKLinear": ("modeling", "HKLinear"),
    "HKQuantizedLinear": ("modeling", "HKQuantizedLinear"),
    "HKTransformerBlock": ("modeling", "HKTransformerBlock"),
    "HKForCausalLM": ("modeling", "HKForCausalLM"),
    "HKForSequenceClassification": ("modeling", "HKForSequenceClassification"),
    "HKForHandwritingRecognition": ("modeling", "HKForHandwritingRecognition"),
    "AutoModel": ("modeling", "AutoModel"),
    "ModelOutput": ("modeling", "ModelOutput"),
    "HKTokenizer": ("tokenizer", "HKTokenizer"),
    "AutoTokenizer": ("tokenizer", "AutoTokenizer"),
    "pipeline": ("pipeline", "pipeline"),
    "BasePipeline": ("pipeline", "BasePipeline"),
    "TextGenerationPipeline": ("pipeline", "TextGenerationPipeline"),
    "SequenceClassificationPipeline": ("pipeline", "SequenceClassificationPipeline"),
    "HandwritingRecognitionPipeline": ("pipeline", "HandwritingRecognitionPipeline"),
    "HKTrainer": ("trainer", "HKTrainer"),
    "HKTrainingArguments": ("trainer", "HKTrainingArguments"),
    "save_file": ("torch", "save_file"),
    "load_file": ("torch", "load_file"),
    "save_model": ("torch", "save_model"),
    "load_model": ("torch", "load_model"),
    "safe_open": ("torch", "safe_open"),
    "load_hk": ("torch", "load_hk"),
    "save_hk": ("torch", "save_hk"),
    "save_sharded_file": ("torch", "save_sharded_file"),
    "load_sharded_file": ("torch", "load_sharded_file"),
    "metadata_set": ("torch", "metadata_set"),
    "ShardedHKFile": ("torch", "ShardedHKFile"),
    "read_remote_hk_header": ("remote", "read_remote_hk_header"),
    "safe_open_remote": ("remote", "safe_open_remote"),
    "RemoteHKFile": ("remote", "RemoteHKFile"),
    "GGUFReaderLight": ("gguf_parser", "GGUFReaderLight"),
    "convert_gguf_to_hk": ("gguf_parser", "convert_gguf_to_hk"),
    "export_hk_to_gguf": ("gguf_parser", "export_hk_to_gguf"),
    "HFArchitectureMapper": ("hf_mapper", "HFArchitectureMapper"),
    "convert_hf_checkpoint": ("hf_mapper", "convert_hf_checkpoint"),
    "permute_hf_to_gguf_rope": ("hf_mapper", "permute_hf_to_gguf_rope"),
    "unpermute_gguf_to_hf_rope": ("hf_mapper", "unpermute_gguf_to_hf_rope"),
    "add_layernorm_offset": ("hf_mapper", "add_layernorm_offset"),
    "sub_layernorm_offset": ("hf_mapper", "sub_layernorm_offset"),
    "split_fused_qkv": ("hf_mapper", "split_fused_qkv"),
    "merge_fused_qkv": ("hf_mapper", "merge_fused_qkv"),
    "pack_moe_experts": ("hf_mapper", "pack_moe_experts"),
    "unpack_moe_experts": ("hf_mapper", "unpack_moe_experts"),
    "UniversalPipeline": ("composite", "UniversalPipeline"),
    "PipelineStage": ("composite", "PipelineStage"),
    "PipelineContext": ("composite", "PipelineContext"),
    "CompositePipeline": ("composite", "CompositePipeline"),
    "ContextWindowManager": ("composite", "ContextWindowManager"),
    "HKWhisperModel": ("composite", "HKWhisperModel"),
    "HKDistilBertModel": ("composite", "HKDistilBertModel"),
    "is_native_available": ("native", "is_native_available"),
    "NativeHKTokenizer": ("native", "NativeHKTokenizer"),
    "NativeHKEngine": ("native", "NativeHKEngine"),
    "convert_safetensors_to_hk": ("native", "convert_safetensors_to_hk"),
    "HardwareMemoryInspector": ("offload", "HardwareMemoryInspector"),
    "LayerMemoryEstimator": ("offload", "LayerMemoryEstimator"),
    "DynamicOffloadPlanner": ("offload", "DynamicOffloadPlanner"),
    "AutoDeviceDispatcher": ("offload", "AutoDeviceDispatcher"),
    "DynamicOOMGuard": ("offload", "DynamicOOMGuard"),
    "DeviceMemoryInfo": ("offload", "DeviceMemoryInfo"),
    "launch_gui": ("gui", "launch_gui"),
    "HKRawWeightStore": ("raw", "HKRawWeightStore"),
    "save_raw": ("raw", "save_raw"),
    "load_raw": ("raw", "load_raw"),
    "save_sharded_raw": ("raw", "save_sharded_raw"),
    "load_sharded_raw": ("raw", "load_sharded_raw"),
    "to_amd_rocm": ("raw", "to_amd_rocm"),
    "to_intel_npu": ("raw", "to_intel_npu"),
    "to_apple_metal": ("raw", "to_apple_metal"),
    "to_nvidia_tensor_core": ("raw", "to_nvidia_tensor_core"),
    "native_detect_hardware": ("native", "native_detect_hardware"),
    "native_get_optimal_alignment": ("native", "native_get_optimal_alignment"),
    "StorageType": ("format", "StorageType"),
    "TileLayout": ("format", "TileLayout"),
    "SparsityType": ("format", "SparsityType"),
    "tile_matrix_16x16": ("format", "tile_matrix_16x16"),
    "untile_matrix_16x16": ("format", "untile_matrix_16x16"),
    "align_forward": ("format", "align_forward"),
    "make_2_4_sparse": ("quantization", "make_2_4_sparse"),
    "pack_2_4": ("quantization", "pack_2_4"),
    "unpack_2_4": ("quantization", "unpack_2_4"),
    "quantize_nf4_dual_mode": ("quantization", "quantize_nf4_dual_mode"),
    "dequantize_nf4_dual_mode": ("quantization", "dequantize_nf4_dual_mode"),
    "quantize_dq8_dual_mode": ("quantization", "quantize_dq8_dual_mode"),
    "dequantize_dq8_dual_mode": ("quantization", "dequantize_dq8_dual_mode"),
    "quantize_dqt": ("quantization", "quantize_dqt"),
    "dequantize_dqt": ("quantization", "dequantize_dqt"),
    "quantize_q4_k": ("quantization", "quantize_q4_k"),
    "dequantize_q4_k": ("quantization", "dequantize_q4_k"),
    "quantize_q8_k": ("quantization", "quantize_q8_k"),
    "dequantize_q8_k": ("quantization", "dequantize_q8_k"),
    "dequantize_q6_k": ("quantization", "dequantize_q6_k"),
    "dequantize_q2_k": ("quantization", "dequantize_q2_k"),
    "ImportanceMatrixCalibrator": ("quantization", "ImportanceMatrixCalibrator"),
    "QUANT_RECIPES": ("quantization", "QUANT_RECIPES"),
    "resolve_quant_type_for_tensor": ("quantization", "resolve_quant_type_for_tensor"),
    "prune_unstructured_magnitude": ("pruning", "prune_unstructured_magnitude"),
    "prune_wanda": ("pruning", "prune_wanda"),
    "prune_structured_2_4": ("pruning", "prune_structured_2_4"),
    "prune_block_sparse": ("pruning", "prune_block_sparse"),
    "prune_structured_l2": ("pruning", "prune_structured_l2"),
    "fine_tune_recovery": ("pruning", "fine_tune_recovery"),
    "LayerSparsitySchedule": ("pruning", "LayerSparsitySchedule"),
    "benchmark_model": ("benchmark", "benchmark_model"),
    "compare_models": ("benchmark", "compare_models"),
    "GrowthGovernor": ("adaptive", "GrowthGovernor"),
    "net2wider_linear": ("adaptive", "net2wider_linear"),
    "net2wider_swiglu": ("adaptive", "net2wider_swiglu"),
    "net2deeper_linear": ("adaptive", "net2deeper_linear"),
    "expand_vocab": ("adaptive", "expand_vocab"),
    "expand_model_width": ("adaptive", "expand_model_width"),
    "protect_base_capacity": ("adaptive", "protect_base_capacity"),
    "AppendixRecord": ("adaptive", "AppendixRecord"),
    "AppendixEntryType": ("adaptive", "AppendixEntryType"),
    "AppendixFlags": ("adaptive", "AppendixFlags"),
    "read_appendix": ("adaptive", "read_appendix"),
    "write_appendix_record": ("adaptive", "write_appendix_record"),
    "rollback_appendix": ("adaptive", "rollback_appendix"),
    "CodeSandbox": ("adaptive", "CodeSandbox"),
    "SelfConversationalEngine": ("adaptive", "SelfConversationalEngine"),
    "SelfTrainingPipeline": ("adaptive", "SelfTrainingPipeline"),
    "AutoModelForCausalLM": ("modeling", "AutoModel"),
    "AutoModelForSequenceClassification": ("modeling", "AutoModel"),
}

# Submodules reachable as `hk.<name>`.
_SUBMODULES = {
    "torch",
    "native",
    "tokenizer",
    "cli",
    "adaptive",
    "benchmark",
    "flax",
    "format",
    "gui",
    "jax",
    "numpy",
    "pruning",
    "quantization",
    "raw",
}


def _explain_missing(e: ImportError, name: str) -> None:
    """Turn a bare `No module named 'torch'` raised inside an hk module into advice about the extra to install."""
    from ._lazy import MissingOptionalDependency

    missing = (e.name or "").split(".")[0]
    if missing in ("torch", "numpy"):
        raise MissingOptionalDependency(
            f"`hk.{name}` needs `{missing}`, which is not installed. Install it with: pip install 'hknt[{missing}]'"
        ) from e


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module, attr = _EXPORTS[name]
        try:
            value = getattr(importlib.import_module(f".{module}", __name__), attr)
        except ImportError as e:
            _explain_missing(e, name)
            raise
        globals()[name] = value
        return value
    if name in _SUBMODULES or not name.startswith("_"):
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            if e.name != f"{__name__}.{name}":
                _explain_missing(e, name)
                raise  # the submodule exists but one of its own imports is missing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_EXPORTS) | set(_SUBMODULES))


__all__ = [
    "HKConfig",
    "AutoConfig",
    "HKPreTrainedModel",
    "HKLinear",
    "HKTransformerBlock",
    "HKForCausalLM",
    "HKForSequenceClassification",
    "HKForHandwritingRecognition",
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForSequenceClassification",
    "ModelOutput",
    "HKTokenizer",
    "AutoTokenizer",
    "pipeline",
    "BasePipeline",
    "TextGenerationPipeline",
    "SequenceClassificationPipeline",
    "HandwritingRecognitionPipeline",
    "HKTrainer",
    "HKTrainingArguments",
    "torch",
    "save_file",
    "load_file",
    "save_model",
    "load_model",
    "safe_open",
    "ShardedHKFile",
    "save_sharded_file",
    "load_sharded_file",
    "metadata_set",
    "numpy",
    "jax",
    "flax",
    "HFArchitectureMapper",
    "convert_hf_checkpoint",
    "UniversalPipeline",
    "PipelineStage",
    "PipelineContext",
    "CompositePipeline",
    "ContextWindowManager",
    "HKWhisperModel",
    "HKDistilBertModel",
    "HKQuantizedLinear",
    "GGUFReaderLight",
    "convert_gguf_to_hk",
    "export_hk_to_gguf",
    "permute_hf_to_gguf_rope",
    "unpermute_gguf_to_hf_rope",
    "add_layernorm_offset",
    "sub_layernorm_offset",
    "split_fused_qkv",
    "merge_fused_qkv",
    "pack_moe_experts",
    "unpack_moe_experts",
    "read_remote_hk_header",
    "safe_open_remote",
    "RemoteHKFile",
    "gui",
    "launch_gui",
    "raw",
    "HKRawWeightStore",
    "save_raw",
    "load_raw",
    "save_sharded_raw",
    "load_sharded_raw",
    "to_amd_rocm",
    "to_intel_npu",
    "to_apple_metal",
    "to_nvidia_tensor_core",
    "native_detect_hardware",
    "native_get_optimal_alignment",
    "HardwareMemoryInspector",
    "LayerMemoryEstimator",
    "DynamicOffloadPlanner",
    "AutoDeviceDispatcher",
    "DynamicOOMGuard",
    "DeviceMemoryInfo",
]
