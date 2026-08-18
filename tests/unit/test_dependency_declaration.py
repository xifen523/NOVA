from pathlib import Path
import re

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def declared_requirement(name: str) -> str:
    text = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"' + re.escape(name) + r'([^";]*)"', text, re.IGNORECASE)
    assert match, "missing dependency declaration: " + name
    return match.group(1)


def test_runtime_and_package_versions_match() -> None:
    import nova

    text = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match
    assert nova.__version__ == match.group(1)
    assert str(Version(nova.__version__)) == nova.__version__


def test_core_dependencies_are_declared_importable_and_functional() -> None:
    import numpy
    import scipy
    import torch
    import yaml
    from scipy.optimize import linear_sum_assignment

    versions = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "PyYAML": yaml.__version__,
        "torch": torch.__version__.split("+")[0],
    }
    for name, version in versions.items():
        specifier = SpecifierSet(declared_requirement(name))
        assert Version(version) in specifier, "{0} {1} violates {2}".format(name, version, specifier)
    rows, columns = linear_sum_assignment(numpy.array([[2.0, 1.0], [1.0, 2.0]]))
    assert rows.tolist() == [0, 1]
    assert columns.tolist() == [1, 0]
    assert torch.isfinite(torch.tensor([1.0])).all().item()
    assert yaml.safe_load("value: 1") == {"value": 1}
