from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_readme_and_docs_local_links_resolve():
    documents = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            destination = (document.parent / target.split("#", 1)[0]).resolve()
            assert destination.exists(), "broken link in {0}: {1}".format(document, target)


def test_public_text_has_no_internal_stage_or_evidence_language():
    roots = [ROOT / name for name in ("README.md", "CITATION.cff", "docs", "nova", "configs", "tools")]
    forbidden = re.compile("|".join((
        chr(68) + r"4[ABC]", chr(68) + r"5[AB]", chr(68) + r"6-R2",
        "Batch " + r"[345]",
        "host" + "-A-" + "reference", "private fusion " + "work" + "tree",
        "current authorized " + "stage", "accepted " + "tag",
        "internal" + "_evidence",
    )))
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".cff"}:
                assert forbidden.search(path.read_text(encoding="utf-8")) is None, str(path)


def test_documented_configs_and_quickstart_inputs_exist():
    for path in (
        "configs/examples/synthetic_v2x.yaml",
        "configs/examples/v2x_example.yaml",
        "tests/fixtures/synthetic_v2x.json",
    ):
        assert (ROOT / path).is_file()
