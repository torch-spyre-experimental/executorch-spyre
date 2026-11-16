# examples/spyre2_mm

spyre2 (Triton/KTIR path) matmul export demo.

See `docs/spyre2-early-access.md` for prerequisites.

## Quick start

```bash
# Stage 2 (device-runnable, sdsc artifact — default):
python examples/spyre2_mm/export_matmul.py --out-dir artifacts-spyre2

# Stage 1 (host-only KTIR structural verification):
python examples/spyre2_mm/export_matmul.py --artifact ktir --out-dir artifacts-spyre2-ktir
```

Writes `spyre2_matmul.pte` plus `a.bin`, `b.bin`, `ref.bin`, and `inputs.json`.
The `ktir` artifact is host-only and not device-runnable on the shipped SDK.
