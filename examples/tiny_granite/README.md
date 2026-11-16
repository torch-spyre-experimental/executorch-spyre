# examples/tiny_granite

Standalone attention-block export.

See `docs/setup-reference.md` for prerequisites.

## Scripts

| Script | Description |
|---|---|
| `export_attention.py` | Export a standalone GQA attention block (useful for delegation debugging). |

## Quick start

```bash
# 1. Build the device runtime (only needed once per pod)
bash scripts/build_device_runtime.sh

# 2. Device-free delegation check
python examples/tiny_granite/export_attention.py --inspect
```

## Model config

Defaults: `--embed-dim 256 --n-heads 4 --n-kv-heads 2 --seq 16`; see `--help` for the rest.
