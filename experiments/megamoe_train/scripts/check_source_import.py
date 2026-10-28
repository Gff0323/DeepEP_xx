"""Check the fixed source import, including unchanged BF16 components.

The expected token hashes come from b8cff148 after the documented path, symbol
and FFTS-argument substitutions, never from a candidate diff. Whitespace and
comments are ignored so the repository formatter can run. This is a scope and
preservation check, not numerical/device qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
PREFIX = MODULE.relative_to(ROOT).as_posix()
SOURCE = "b8cff14854d12aef77b6ae54b5890123759b4d28"
TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\w+|[^\s]', re.S)


def token_hash(text):
    # The two aggregate-header guards exclude unrelated, unused HSTU templates
    # in the target CATLASS. Used BF16 MMAD specializations are unchanged.
    text = re.sub(
        r"^#(?:define|undef) CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_(?:QK|PV)_SPLIT_ROW_TLA_HPP\s*$", "", text, flags=re.M
    )
    text = text.replace("\\\n", "")
    tokens = [m.group() for m in TOKEN.finditer(text) if not m.group().startswith(("//", "/*"))]
    return hashlib.sha256("\0".join(tokens).encode()).hexdigest()


def check(ref=None, staged=False):
    def read(relative):
        if ref or staged:
            spec = f":{PREFIX}/{relative}" if staged else f"{ref}:{PREFIX}/{relative}"
            return subprocess.check_output(["git", "-C", str(ROOT), "show", spec]).decode("utf-8")
        return (MODULE / relative).read_text(encoding="utf-8")

    manifest = json.loads(read("upstream.json"))
    if manifest["commit"] != SOURCE:
        raise RuntimeError("Source revision changed; review a new complete migration scope")
    checked = 0
    for entry in manifest["files"]:
        content = read(entry["target"])
        if "preserved_token_sha256" in entry:
            if token_hash(content) != entry["preserved_token_sha256"]:
                raise RuntimeError(f"BF16 source semantics changed: {entry['target']}")
            checked += 1
    for relative in manifest["adapted_required_files"]:
        read(relative)
    if checked != 52:
        raise RuntimeError(f"Incomplete fixed source scope: expected 52 preserved C++ files, got {checked}")
    print(
        f"PASS: {checked} preserved C++ files; {len(manifest['files'])} imports and runtime/API/build adapters present"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--ref")
    args = parser.parse_args()
    check(args.ref, args.staged)
