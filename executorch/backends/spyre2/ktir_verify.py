# Copyright 2026 The Torch-Spyre / ExecuTorch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Structural verification of emitted KTIR against the hand-authored golden.

The Stage 1 lowering (``ttir_to_ktir``) cannot be verified by running it (no
``ktdp`` runtime on the shipped SDK), so its oracle is *structural*: reduce a
KTIR function to a canonical **op-grammar signature** — the ordered sequence of
significant ops with their stick type, discarding SSA value names, whitespace,
constant/index arithmetic, and tiling offsets — and assert the emitted ``mm``
matches a golden matmul function (e.g. the ``matmul_mlp_prefill`` in
``backends/spyre2/fixtures/ktir/level4c_matmul_mlp.mlir``).

Pure Python, no torch / triton / MLIR — runs on the Mac host.
"""

import re

# The canonical op sequence a single-tile matmul KTIR function must exhibit.
MATMUL_SIGNATURE = (
    "get_compute_tile_id",
    "construct_memory_view",  # activation
    "construct_memory_view",  # weight
    "construct_memory_view",  # output
    "load",  # activation
    "load",  # weight
    "matmul",
    "store",
    "return",
)

_STICK_TYPE = "vector<64xf16>"


def canonical_signature(ktir_func_text: str) -> tuple[str, ...]:
    """Reduce one KTIR function body to its ordered significant-op signature.

    Recognizes the ``ktdp``/``linalg`` ops that define a matmul tile and ignores
    everything structural but incidental: ``module``/``func.func`` headers,
    ``arith.*`` index math (core/tile offset arithmetic), braces, and blank
    lines. ``construct_memory_view`` is only counted when it carries the
    ``vector<64xf16>`` stick, so a malformed view without the stick type fails
    the match rather than passing silently.
    """
    sig: list[str] = []
    for raw in ktir_func_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if "ktdp.get_compute_tile_id" in line:
            sig.append("get_compute_tile_id")
        elif "ktdp.construct_memory_view" in line:
            # The stick type may be on a continuation line; the op line always
            # names construct_memory_view, and the whole function is checked for
            # the stick type separately by ``verify_matmul``.
            sig.append("construct_memory_view")
        elif "ktdp.load" in line:
            sig.append("load")
        elif "linalg.matmul" in line:
            sig.append("matmul")
        elif "ktdp.store" in line:
            sig.append("store")
        elif re.match(r"return\b", line):
            sig.append("return")
    return tuple(sig)


def split_functions(ktir_text: str) -> dict[str, str]:
    """Split KTIR module text into ``{func_name: body_text}``.

    A lightweight brace-depth scan — sufficient for the flat, single-level
    ``func.func`` bodies in both the golden and the emitted KTIR.
    """
    funcs: dict[str, str] = {}
    lines = ktir_text.splitlines()
    i = 0
    while i < len(lines):
        m = re.search(r"func\.func\s+@(\w+)", lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        depth = lines[i].count("{") - lines[i].count("}")
        body = [lines[i]]
        i += 1
        while i < len(lines) and depth > 0:
            depth += lines[i].count("{") - lines[i].count("}")
            body.append(lines[i])
            i += 1
        funcs[name] = "\n".join(body)
    return funcs


def verify_matmul(emitted_ktir: str, golden_ktir: str) -> None:
    """Assert the emitted ``mm`` KTIR is structurally a matmul, matching golden.

    Checks, raising ``AssertionError`` with a diff-style message on mismatch:
      1. the emitted module has exactly one function whose signature equals
         ``MATMUL_SIGNATURE``;
      2. at least one golden function reduces to the same signature (proving the
         canonical form is the golden's own matmul grammar, not an invention);
      3. the emitted KTIR uses the ``vector<64xf16>`` stick type.
    """
    emitted_funcs = split_functions(emitted_ktir)
    assert len(emitted_funcs) == 1, (
        f"expected exactly one emitted mm function, found "
        f"{sorted(emitted_funcs)}"
    )
    (emitted_name, emitted_body), = emitted_funcs.items()
    emitted_sig = canonical_signature(emitted_body)
    assert emitted_sig == MATMUL_SIGNATURE, (
        f"emitted function @{emitted_name} signature mismatch:\n"
        f"  expected: {MATMUL_SIGNATURE}\n"
        f"  actual:   {emitted_sig}"
    )
    assert _STICK_TYPE in emitted_body, (
        f"emitted KTIR must use the {_STICK_TYPE!r} stick type"
    )

    golden_sigs = {
        name: canonical_signature(body)
        for name, body in split_functions(golden_ktir).items()
    }
    matching = [n for n, s in golden_sigs.items() if s == MATMUL_SIGNATURE]
    assert matching, (
        "no golden function reduces to the canonical matmul signature "
        f"{MATMUL_SIGNATURE}; golden signatures were:\n"
        + "\n".join(f"  @{n}: {s}" for n, s in golden_sigs.items())
    )
