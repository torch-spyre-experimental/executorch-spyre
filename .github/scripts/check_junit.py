# Copyright 2026 The ExecuTorch-Spyre Authors.
# Licensed under the Apache License, Version 2.0.
"""Fail CI when a JUnit report shows fewer passes or more skips than expected.

The host suites ``importorskip`` everything Spyre-specific, so a broken
install (e.g. torch failing to import) can turn the whole run into skips and
still exit 0. This guard catches that by checking the counts, not just the
exit status. It also writes a one-line summary to ``$GITHUB_STEP_SUMMARY``.

    python .github/scripts/check_junit.py REPORT.xml --label spyre \
        --min-passed 73 --max-skipped 4
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def _totals(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.iter("testsuite")
    tests = failures = errors = skipped = 0
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", s.get("disabled", 0)))
    return tests - failures - errors - skipped, failures + errors, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report")
    ap.add_argument("--label", default="tests")
    ap.add_argument("--min-passed", type=int, required=True)
    ap.add_argument("--max-skipped", type=int, default=0)
    args = ap.parse_args()

    passed, failed, skipped = _totals(args.report)
    problems = []
    if failed:
        problems.append(f"{failed} failed/errored")
    if passed < args.min_passed:
        problems.append(f"passed {passed} < expected {args.min_passed}")
    if skipped > args.max_skipped:
        problems.append(f"skipped {skipped} > allowed {args.max_skipped}")

    status = "FAIL: " + "; ".join(problems) if problems else "ok"
    line = (
        f"{args.label}: {passed} passed, {failed} failed, {skipped} skipped — {status}"
    )
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"- {'❌' if problems else '✅'} {line}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
