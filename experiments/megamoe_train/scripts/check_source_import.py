"""Check the declared source revision and required imported files."""

import argparse
import json
from pathlib import Path
import subprocess

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
PREFIX = MODULE.relative_to(ROOT).as_posix()
SOURCE = "b8cff14854d12aef77b6ae54b5890123759b4d28"


def check(ref=None, staged=False):
    def read(relative):
        if ref or staged:
            spec = f":{PREFIX}/{relative}" if staged else f"{ref}:{PREFIX}/{relative}"
            return subprocess.check_output(["git", "-C", str(ROOT), "show", spec]).decode("utf-8")
        return (MODULE / relative).read_text(encoding="utf-8")

    manifest = json.loads(read("upstream.json"))
    if manifest["commit"] != SOURCE:
        raise RuntimeError("Source revision changed; review a new complete migration scope")
    for entry in manifest["files"]:
        read(entry["target"])
    for relative in manifest["adapted_required_files"]:
        read(relative)
    print(
        f"PASS: source {manifest['commit'][:8]}; {len(manifest['files'])} imports and runtime/API/build adapters present"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--ref")
    args = parser.parse_args()
    check(args.ref, args.staged)
