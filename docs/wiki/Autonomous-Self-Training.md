# Self-Training Toolkit (`hk.adaptive`)

> **Status: experimental research code, Python only, not part of the inference engine.** This page documents building blocks for a "generate, test, learn from what passed" loop. They are components you wire together, not a turnkey autonomous system: you supply the function that produces solutions and the function that performs a training step. No result showing that the loop improves a model is included or claimed.
>
> **Security first:** `CodeSandbox` runs model-written Python. In its default mode it is **not a security boundary** (details below). Do not run untrusted generated code outside a container or VM you are prepared to lose.

Source: `python/hk/adaptive/` (`self_training.py`, `self_conversation.py`, `code_eval.py`, `expansion_evaluator.py`, `self_play.py`). These operate on the Python `HKForCausalLM` family (see [Training and Fine-Tuning](Training-and-Fine-Tuning)), not on the Zig engine.

---

## Pieces

| Component | What it actually does |
| :--- | :--- |
| `CodeSandbox` (alias `SandboxExecutor`) | Syntax-checks code with `ast.parse`, then runs it plus optional test calls in a child Python process with a wall-clock timeout, and parses a JSON result block from stdout into an `EvalResult` (success, pass rate, stdout/stderr, timing). Optional Docker mode (`use_docker=True`, requires the Docker CLI and the image already pulled): `--network none`, `--memory`, `--cpus`. |
| `SelfConversationalEngine` | Runs a propose/think/test/reflect loop around caller-supplied functions. Extracts `<think>...</think>` reasoning and a code block from a response, runs the code in the sandbox, and on failure calls a reflector for a retry (up to `max_reflection_steps`). It formats successful traces as ChatML-style training pairs. |
| `ExpansionEvaluator` | Heuristics over numbers you provide (diagnostic loss/perplexity, pass rate, missing domain keywords) that decide whether to recommend vocabulary and/or width growth, subject to a `GrowthGovernor` budget. It does not probe the model itself. |
| `SelfTrainingPipeline` | Coordinates the above for one generation: for each curriculum task, run a self-dialogue, count passes, call your `optimizer_step_fn` on the passing dialogues, and append a `code_eval` record to the model's appendix. Optional autonomous expansion hooks into `ExpansionEvaluator` and `growth.py`. |
| `SPINLoss`, `SelfPlayEvolutionEngine`, `LoRAAdapter` (`self_play.py`) | A SPIN-style preference loss between current and previous-generation log-probabilities, a LoRA adapter module, and a loop that records adapters to the appendix and can roll back on regression. |

### `CodeSandbox` limits

- **Subprocess mode (default):** only a timeout. There is no import filter, no memory cap, no filesystem or network isolation; the code runs as your user. The earlier description of AST-based import blocking and memory limits was wrong. `ast.parse` is used only to detect syntax errors.
- **Docker mode:** network disabled, memory and CPU limits applied. If Docker is unavailable or the image is missing it falls back silently to the unsandboxed subprocess mode; check `sandbox.is_docker_active`.
- Only Python is executed. A `target_language` of other languages affects prompt formatting, not execution.

---

## Using the pipeline

```python
from hk.adaptive import (
    SelfTrainingPipeline, SelfTrainingCurriculum, SelfConversationalEngine, CodeSandbox,
)

curriculum = SelfTrainingCurriculum(
    domain_name="Algorithms",
    target_language="python",
    syntax_keywords=["def", "return", "yield"],
    training_tasks=[{
        "id": "palindrome",
        "prompt": "Write is_palindrome(s).",
        "test_cases": [{"call": "is_palindrome('racecar')", "expected": True}],
    }],
)

engine = SelfConversationalEngine(sandbox=CodeSandbox(timeout_sec=2.0, use_docker=True))
pipeline = SelfTrainingPipeline(model=model, hk_file_path="model.hk", conversational_engine=engine)

def solution_generator_fn(prompt: str) -> str:
    ...   # your model sampling code; return text containing <think>..</think> and a code block

def optimizer_step_fn(dialogues, model) -> float:
    ...   # your training step on the passing dialogues; return the loss

report = pipeline.train_generation(
    curriculum,
    solution_generator_fn=solution_generator_fn,
    optimizer_step_fn=optimizer_step_fn,
)
print(report.pass_rate, report.successful_dialogues, report.train_loss)
```

`GenerationReport` fields: `generation`, `train_loss`, `pass_rate`, `expansion_occurred`, `expansion_details`, `evaluated_tasks`, `successful_dialogues`. Test cases use `call`/`expected` (and optional `desc`) keys. Check constructor arguments in `self_training.py` before relying on optional ones.

---

## Caveats

- The "autonomy" is the loop structure; model quality, sampling and the training step are yours.
- Pass rate is measured only on the tests you write; models can pass weak tests without being correct.
- Training on a model's own verified outputs can reinforce mistakes the tests miss. Nothing here guards against that beyond rehearsal and the appendix rollback tools.
- Appendix records are written by `append_record`; see [In-Container Version Lineage](In-Container-Version-Lineage) for what that does and does not give you.
