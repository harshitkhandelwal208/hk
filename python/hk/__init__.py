"""
HK: Next-Generation AI Framework
Native-first, dual-mode quantization, adaptive topology growth, and ultra-portable inference.
"""

from .config import HKConfig, AutoConfig
from .modeling import (
    HKPreTrainedModel,
    HKLinear,
    HKQuantizedLinear,
    HKTransformerBlock,
    HKForCausalLM,
    HKForSequenceClassification,
    HKForHandwritingRecognition,
    AutoModel,
    ModelOutput,
)
from .tokenizer import HKTokenizer, AutoTokenizer
from .pipeline import pipeline, BasePipeline, TextGenerationPipeline, SequenceClassificationPipeline, HandwritingRecognitionPipeline
from .trainer import HKTrainer, HKTrainingArguments
from .torch import save_file, load_file, save_model, load_model, safe_open, load_hk, save_hk, save_sharded_file, load_sharded_file, metadata_set, ShardedHKFile
from .remote import read_remote_hk_header, safe_open_remote, RemoteHKFile
from .gguf_parser import GGUFReaderLight, convert_gguf_to_hk, export_hk_to_gguf
from .hf_mapper import (
    HFArchitectureMapper,
    convert_hf_checkpoint,
    permute_hf_to_gguf_rope,
    unpermute_gguf_to_hf_rope,
    add_layernorm_offset,
    sub_layernorm_offset,
    split_fused_qkv,
    merge_fused_qkv,
    pack_moe_experts,
    unpack_moe_experts,
)
from .composite import (
    UniversalPipeline,
    PipelineStage,
    PipelineContext,
    CompositePipeline,
    ContextWindowManager,
    HKWhisperModel,
    HKDistilBertModel,
)
from .native import (
    is_native_available,
    NativeHKTokenizer,
    NativeHKEngine,
    convert_safetensors_to_hk,
)
from .offload import (
    HardwareMemoryInspector,
    LayerMemoryEstimator,
    DynamicOffloadPlanner,
    AutoDeviceDispatcher,
    DynamicOOMGuard,
    DeviceMemoryInfo,
)
from . import format
from . import quantization
from . import pruning
from . import benchmark
from . import adaptive
from . import numpy
from . import raw
from . import gui
from .gui import launch_gui
from .raw import (
    HKRawWeightStore,
    save_raw,
    load_raw,
    save_sharded_raw,
    load_sharded_raw,
    to_amd_rocm,
    to_intel_npu,
    to_apple_metal,
    to_nvidia_tensor_core,
)
from .native import (
    native_detect_hardware,
    native_get_optimal_alignment,
)
try:
    from . import jax
    from . import flax
except ImportError:
    pass

from .format import (
    StorageType,
    TileLayout,
    SparsityType,
    tile_matrix_16x16,
    untile_matrix_16x16,
    align_forward,
)
from .quantization import (
    make_2_4_sparse,
    pack_2_4,
    unpack_2_4,
    quantize_nf4_dual_mode,
    dequantize_nf4_dual_mode,
    quantize_dq8_dual_mode,
    dequantize_dq8_dual_mode,
    quantize_dqt,
    dequantize_dqt,
    quantize_q4_k,
    dequantize_q4_k,
    quantize_q8_k,
    dequantize_q8_k,
    dequantize_q6_k,
    dequantize_q2_k,
    ImportanceMatrixCalibrator,
    QUANT_RECIPES,
    resolve_quant_type_for_tensor,
)
from .pruning import (
    prune_unstructured_magnitude,
    prune_wanda,
    prune_structured_2_4,
    prune_block_sparse,
    prune_structured_l2,
    fine_tune_recovery,
    LayerSparsitySchedule,
)
from .benchmark import benchmark_model, compare_models
from .adaptive import (
    GrowthGovernor,
    net2wider_linear,
    net2wider_swiglu,
    net2deeper_linear,
    expand_vocab,
    expand_model_width,
    protect_base_capacity,
    AppendixRecord,
    AppendixEntryType,
    AppendixFlags,
    read_appendix,
    write_appendix_record,
    rollback_appendix,
    CodeSandbox,
    SelfConversationalEngine,
    SelfTrainingPipeline,
)

# Aliases for Hugging Face muscle memory
AutoModelForCausalLM = AutoModel
AutoModelForSequenceClassification = AutoModel

__version__ = "1.1.1"

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
