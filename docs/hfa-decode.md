<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Export and run Granite

Export a Hugging Face Granite checkpoint to a Spyre `.pte` and generate text from it with
`spyre_llm_runner`. (Mistral and Ministral are dispatched the same way, from the checkpoint's
`config.json`.)

Start from the shell where [setup.md](setup.md) finished. Its variables (`ET_HOME`, `CKPT`,
`RUN_BUILD`, `PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`, ...) must still be set and
`$RUN_BUILD/spyre_llm_runner` must exist. Building the runner is in [setup.md](setup.md); how the
export and the runtime work is in [hfa-design.md](hfa-design.md).

## 1. Set up the shell

```bash
cd "$ET_HOME"
export HF_ADAPTERS_ROOT=/path/to/hf-adapters
export OUT_DIR=/tmp/et-decode-llm
```

## 2. Export

Use [`export_hfa.py`](../executorch/backends/spyre/export_hfa.py). "Full depth" means all of
the checkpoint's layers, i.e. no `--num-layers` flag.

### 2.1 Granite: `--llm-wrapper`

```bash
python executorch/backends/spyre/export_hfa.py \
    --ckpt "$CKPT" \
    --mode decode \
    --llm-wrapper \
    --seq 128 \
    --stage pte \
    --out-dir "$OUT_DIR"
```

**Check:** `$OUT_DIR/spyre_hfa_decode_llm.pte` exists.

### 2.2 Reference path: `--mode prefill --kv buffer`

```bash
python executorch/backends/spyre/export_hfa.py \
    --ckpt "$CKPT" \
    --mode prefill --kv buffer \
    --rope gather --norm orig \
    --seq 64 \
    --stage pte \
    --out-dir /tmp/et-prefill
```

**Check:** `/tmp/et-prefill/spyre_hfa_prefill.pte` exists.

## 3. Run

```bash
"$RUN_BUILD/spyre_llm_runner" \
    --pte "$OUT_DIR/spyre_hfa_decode_llm.pte" \
    --tokenizer "$CKPT/tokenizer.json" \
    --prompt "Write a short poem about computing:" \
    --max-new 32
```

**Expected output:** the runner prints `RUNTIME_PATH mode=components`, a `tokenizer: vocab_size=...`
line, a token trace, and finally `GENERATED_TEXT:[...]`.

A model exported with `--num-layers 2` (a quick smoke test) produces degenerate output such as
`GENERATED_TEXT:[:::::::::::::::::::::::::::::::]`. That is expected for 2 layers and is not an error.

If the build or the import fails, go back to the failures table in [setup.md](setup.md).
