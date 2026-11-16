# examples/mlp

Exports a fp16 MLP (with optional mixed-dtype inputs) to `.pte`.

See the top-level README for environment setup.

## Scripts

| Script | Description |
|---|---|
| `export_mlp.py` | `Linear(64,128)->ReLU->Linear(128,64)` fully delegated to Spyre. Pass `--buckets 1,4,16` for a dynamic-batch `.pte`. |
| `export_mixed.py` | `relu(sin(fc(x)))` — minimal CPU-fallback demo: `fc` and `relu` delegate to Spyre, `sin` runs on the portable CPU kernel. Requires the runner built with `SPYRE_RUNNER_WITH_PORTABLE_OPS=ON`. |

## Quick start

```bash
python examples/mlp/export_mlp.py --out-dir artifacts
python examples/mlp/export_mixed.py --out-dir artifacts-mixed --inspect  # device-free check
```
