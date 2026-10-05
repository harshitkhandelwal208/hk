# In-Container Version Lineage (Appendix)

> **Status:** the appendix is a working, tested feature of the container: append-only records, a SHA-256 parent chain, listing, and truncating rollback. It is a *record-keeping* mechanism. The inference engine does not read or apply appendix records (a model runs from its base tensors only), and the chain is not a signature: it detects accidental or partial edits to the appendix, not a determined attacker.

Binary layout of records is in [Format Specification](Format-Specification) section 8.

---

## What it does

Records are appended after the base tensors; the base weights are never rewritten. Each record carries a name, a target (for example the tensor an adapter applies to), a generation number, a timestamp, four metrics (`loss`, `accuracy`, `pass_rate`, `custom`), a payload, and the SHA-256 of the previous record's `name ‖ target ‖ data`. Adding a record costs a write of that record's size, not of the model: `appendRecordToFile` writes at end of file and patches the header only for the first record.

Entry types: `lora_adapter`, `delta_patch`, `new_layer`, `code_eval`, `kv_cache_sink`, `topology_head`. The training tools in `python/hk/adaptive/` write `lora_adapter`/`code_eval`/`topology_head` records; see [Training and Fine-Tuning](Training-and-Fine-Tuning).

---

## CLI

```bash
hk appendix model.hk            # list records and report whether the lineage chain verifies
hk rollback model.hk 2          # drop all records with generation > 2
```

`hk appendix` prints the entry count, `Cryptographic Lineage Valid: true|false`, and a table with index, type, generation, name, target, accuracy and pass rate per record.

### Rollback truncates

`rollbackToFile` scans records in order, keeps those with `generation <= N`, and **truncates the file** after the last kept record (with `N = 0` it also clears `appendix_offset` and the `HAS_APPENDIX` flag). Consequences:

- It is fast (a truncate, no data copy) and does not touch the base weights.
- It is **destructive**: the removed records are gone and cannot be restored from the file. Copy the file first if you may want them back.
- It assumes records are stored in increasing generation order and stops at the first record above the target.

---

## Verifying the chain

`AppendixReader.verifyLineage()` (also shown by `hk appendix`) checks, for each record after the first, that its `parent_hash` equals SHA-256 of the previous record's `name ‖ target ‖ data` (the payload-only hash used by older files is also accepted). It reports a bool; it does not say which record failed.

What this does and does not protect:

- Detects: edits to an earlier record's name, target or payload; deleted or reordered middle records.
- Does not cover: the base tensors, the header, or metadata. Changing weights does not invalidate the chain.
- Does not authenticate: anyone who can rewrite the file can recompute the hashes. There are no signatures.
- The per-record `data_crc32` field is reserved and currently always `0`.

---

## Python

```python
from hk.adaptive.appendix import AppendixManager, read_appendix

mgr = AppendixManager("model.hk")

mgr.append_lora_checkpoint(
    name="math-lora-v1", target="blk.0.attn_q.weight", generation=1,
    adapter_bytes=blob, metrics={"loss": 1.42, "accuracy": 0.485},
)

for r in mgr.get_records():
    print(r.generation, r.entry_type.name, r.name, r.metrics.loss)

print("chain valid:", mgr.verify())
mgr.rollback(1)          # same truncating semantics as `hk rollback`
```

Module-level helpers: `read_appendix`, `append_record`, `rollback_appendix`, `verify_lineage`, `compute_parent_hash`. The adaptive training loops (`HKTrainer`, the self-training pipeline) call `append_record` with metrics at checkpoints.

---

## Limitations

- No tool applies appended adapters or deltas to produce a merged model for the engine; merging is up to your own code.
- Appending is not atomic. A crash mid-append can leave a truncated final record, which readers report as `TruncatedRecord`.
- `hk metadata set` refuses to move metadata to the end of a file that has an appendix (the appendix runs to EOF); it succeeds only when the new metadata fits in the existing padding.
