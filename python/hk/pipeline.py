"""
HK Pipeline
Unified high-level task abstraction for inference pipelines.
"""

import sys
import types
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import torch
import numpy as np

from .config import HKConfig, AutoConfig
from .modeling import AutoModel, HKPreTrainedModel, HKForCausalLM, HKForSequenceClassification, HKForHandwritingRecognition
from .tokenizer import AutoTokenizer, HKTokenizer


class BasePipeline:
    def __init__(self, model: HKPreTrainedModel, tokenizer: Optional[HKTokenizer] = None, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TextGenerationPipeline(BasePipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        **kwargs,
    ) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        results = []
        for p in prompts:
            input_ids = self.tokenizer.encode(p, add_special_tokens=True)
            # Remove eos token if present to continue generation
            if input_ids and input_ids[-1] == self.tokenizer.eos_token_id:
                input_ids = input_ids[:-1]

            inp_tensor = torch.tensor([input_ids], dtype=torch.long)
            gen_ids = self.model.generate(
                inp_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )[0].tolist()

            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append({"generated_text": gen_text})
        return results


class SequenceClassificationPipeline(BasePipeline):
    def __call__(self, text: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        results = []
        for t in texts:
            inputs = self.tokenizer(t, return_tensors="pt")
            with torch.no_grad():
                out = self.model(inputs["input_ids"])
                probs = torch.softmax(out.logits, dim=-1)[0]
                pred_class = int(torch.argmax(probs).item())
                confidence = float(probs[pred_class].item())

            results.append({
                "label": f"LABEL_{pred_class}",
                "score": round(confidence, 4),
            })
        return results


class HandwritingRecognitionPipeline(BasePipeline):
    def __call__(self, image: Any, **kwargs) -> List[Dict[str, Any]]:
        # Accept numpy array or file path
        if isinstance(image, (str, Path)):
            # Load basic image or synthetic
            img_tensor = torch.zeros(1, 1, 32, 128)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[None, None, ...]
            elif image.ndim == 3:
                image = image[None, ...]
            img_tensor = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            img_tensor = image if image.ndim == 4 else image.unsqueeze(0)
        else:
            img_tensor = torch.zeros(1, 1, 32, 128)

        with torch.no_grad():
            out = self.model(img_tensor)
            # Logits: [batch, seq_len, num_classes]
            preds = torch.argmax(out.logits, dim=-1)[0].tolist()

        text = "".join(chr(p % 26 + ord('a')) for p in preds if p > 0)
        return [{"transcription": text, "logits": out.logits}]


def pipeline(
    task: str,
    model: Optional[Union[str, Path, HKPreTrainedModel]] = None,
    tokenizer: Optional[Union[str, Path, HKTokenizer]] = None,
    device: str = "cpu",
    **kwargs,
) -> BasePipeline:
    """Instantiate a task-specific pipeline."""
    task = task.lower().replace("_", "-")

    # Resolve model
    if isinstance(model, (str, Path)):
        if str(device).lower() in ("auto", "dynamic", "dynamic_offload"):
            model_obj = AutoModel.from_pretrained(model, device_map="auto", **kwargs)
        else:
            model_obj = AutoModel.from_pretrained(model, device=device, **kwargs)
    elif isinstance(model, HKPreTrainedModel):
        if str(device).lower() in ("auto", "dynamic", "dynamic_offload"):
            model.to_dynamic_offload()
        model_obj = model
    else:
        # Default placeholder model
        cfg = HKConfig(model_type="causal_lm" if "gen" in task else "sequence_classification")
        if str(device).lower() in ("auto", "dynamic", "dynamic_offload"):
            model_obj = HKForCausalLM(cfg, device="cpu").to_dynamic_offload()
        else:
            model_obj = HKForCausalLM(cfg, device=device)

    # Resolve tokenizer
    if isinstance(tokenizer, (str, Path)):
        tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer)
    elif isinstance(tokenizer, HKTokenizer):
        tokenizer_obj = tokenizer
    else:
        tokenizer_obj = AutoTokenizer.from_pretrained(model if isinstance(model, (str, Path)) else ".")

    if task in ("text-generation", "causal-lm", "lm"):
        return TextGenerationPipeline(model=model_obj, tokenizer=tokenizer_obj, device=device)
    elif task in ("sequence-classification", "sentiment-analysis", "text-classification"):
        return SequenceClassificationPipeline(model=model_obj, tokenizer=tokenizer_obj, device=device)
    elif task in ("handwriting-recognition", "hwr", "ocr", "image-to-text"):
        return HandwritingRecognitionPipeline(model=model_obj, tokenizer=tokenizer_obj, device=device)
    elif task in ("composite", "speech-analysis-generation", "speech-to-text-to-generation"):
        from .composite import CompositePipeline, ContextWindowManager, HKWhisperModel, HKDistilBertModel
        whisper_model = kwargs.get("whisper_model", None)
        analyzer_model = kwargs.get("analyzer_model", None)
        llm_model = kwargs.get("llm_model", model_obj if isinstance(model_obj, HKForCausalLM) else None)
        if llm_model is None:
            llm_model = HKForCausalLM(
                HKConfig(model_type="causal_lm", hidden_size=256, num_hidden_layers=2, num_attention_heads=4),
                device=device,
            )
        if whisper_model is None:
            whisper_model = HKWhisperModel(device=device)
        if analyzer_model is None:
            analyzer_model = HKDistilBertModel(device=device)
        ctx_mgr = kwargs.get("context_manager", ContextWindowManager())
        return CompositePipeline(
            whisper_model=whisper_model,
            analyzer_model=analyzer_model,
            llm_model=llm_model,
            tokenizer=tokenizer_obj,
            context_manager=ctx_mgr,
            device=device,
        )
    elif task in ("universal", "multi-modal", "dag"):
        from .composite import UniversalPipeline, PipelineStage, ContextWindowManager
        stages = kwargs.get("stages", [])
        ctx_mgr = kwargs.get("context_manager", ContextWindowManager())
        return UniversalPipeline(stages=stages, tokenizer=tokenizer_obj, context_manager=ctx_mgr, device=device)
    elif task in ("vision-analysis-generation", "vision-text-generation", "ocr-to-generation"):
        from .composite import UniversalPipeline
        return UniversalPipeline.vision_analysis_generation(
            ocr_model=kwargs.get("ocr_model"),
            analyzer_model=kwargs.get("analyzer_model"),
            llm_model=kwargs.get("llm_model", model_obj if isinstance(model_obj, HKForCausalLM) else None),
            tokenizer=tokenizer_obj,
            context_manager=kwargs.get("context_manager"),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported pipeline task: {task}")


class _CallableModule(types.ModuleType):
    """`hk.pipeline(...)` is the factory function, but `hk.pipeline` is also this submodule. Python rebinds the package
    attribute to the module whenever the submodule is first imported, which can happen after the function was bound,
    so the module forwards calls to the function and both spellings work in any import order."""

    def __call__(self, *args, **kwargs):
        return pipeline(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
