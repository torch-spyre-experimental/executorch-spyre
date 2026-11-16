# examples/matmul

Exports a 64×64 fp16 matmul to `.pte` and runs it on the Spyre.

See the top-level README for environment setup.

## Quick start

```bash
python examples/matmul/export_matmul.py --out-dir artifacts
```

Writes `artifacts/spyre_matmul.pte` plus `a.bin`, `b.bin`, `ref.bin`, and `inputs.json`.

Pass `--buckets 128,256,512` to compile a multi-bucket dynamic-M `.pte` (one JobPlan per bucket).
