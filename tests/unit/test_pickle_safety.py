import os
import pickle
from pathlib import Path

from nova.runtime.pickle_safety import inspect_pickle_static


class UnsafeFixture:
    def __reduce__(self):
        return os.system, ("never executed",)


def test_pickle_inspector_never_approves_load(tmp_path: Path) -> None:
    path = tmp_path / "fixture.pkl"
    path.write_bytes(pickle.dumps({"safe": [1, 2, 3]}, protocol=4))
    report = inspect_pickle_static(path)
    assert not report.automated_load_approved
    assert report.persistent_id_count == 0


def test_pickle_inspector_reports_dangerous_global_without_execution(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pkl"
    path.write_bytes(pickle.dumps(UnsafeFixture(), protocol=2))
    report = inspect_pickle_static(path)
    assert any("system" in item for item in report.direct_globals)
    assert report.reduce_count == 1
