#!/usr/bin/env python3
"""Build and verify deterministic public-release integrity metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from nova.release_hash import (
    metadata_tree_sha256,
    tree_sha256,
    tree_sha256_from_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    hashes = {
        "directory": tree_sha256(args.release_root),
        "manifest": tree_sha256_from_manifest(args.manifest),
        "metadata": metadata_tree_sha256(args.metadata),
    }
    if len(set(hashes.values())) != 1:
        for source, digest in hashes.items():
            print(f"{source}={digest}")
        return 1
    print(f"TREE_SHA256={hashes['directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
