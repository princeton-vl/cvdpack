import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def load_verify_tolerant() -> ModuleType:
    path = Path(__file__).parent / "integration_test" / "verify_tolerant.py"
    spec = importlib.util.spec_from_file_location("verify_tolerant", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_reports_numeric_error_above_tolerance(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.save(blessed / "depth.npy", np.array([1.0], dtype=np.float32))
    np.save(actual / "depth.npy", np.array([1.1], dtype=np.float32))

    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == ["depth.npy: max error 0.10000002384185791 exceeds 0.01"]


def test_compare_handles_npz_string_fields(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    names = np.array([b"chair", b"table"], dtype="|S63")
    np.savez(blessed / "object-data.npz", object_name=names)
    np.savez(actual / "object-data.npz", object_name=names)

    assert load_verify_tolerant().compare(blessed, actual, 0.01) == []

    np.savez(actual / "object-data.npz", object_name=names[::-1])
    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == ["object-data.npz: object_name: values differ"]


def test_compare_reports_unsigned_error_without_wraparound(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.save(blessed / "seg.npy", np.array([0], dtype=np.uint16))
    np.save(actual / "seg.npy", np.array([1], dtype=np.uint16))

    problems = load_verify_tolerant().compare(blessed, actual, 0.5)

    assert problems == ["seg.npy: max error 1.0 exceeds 0.5"]


def test_compare_rejects_all_nan_output(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.save(blessed / "depth.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.save(actual / "depth.npy", np.full(3, np.nan, dtype=np.float32))

    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == ["depth.npy: nan/inf in different positions"]


def test_compare_checks_values_beside_a_blessed_nan(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.save(blessed / "depth.npy", np.array([1.0, np.nan, 3.0], dtype=np.float32))
    np.save(actual / "depth.npy", np.array([1.0, np.nan, 999.0], dtype=np.float32))

    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == ["depth.npy: max error 996.0 exceeds 0.01"]


def test_compare_accepts_matching_nan_positions(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    values = np.array([1.0, np.nan, np.inf, 3.0], dtype=np.float32)
    np.save(blessed / "depth.npy", values)
    np.save(actual / "depth.npy", values.copy())

    assert load_verify_tolerant().compare(blessed, actual, 0.01) == []


def test_compare_rejects_inf_swapped_for_nan(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.save(blessed / "depth.npy", np.array([np.inf], dtype=np.float32))
    np.save(actual / "depth.npy", np.array([np.nan], dtype=np.float32))

    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == ["depth.npy: nan/inf values differ"]


def test_compare_rejects_empty_blessed_tree(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.save(actual / "depth.npy", np.array([1.0], dtype=np.float32))

    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == [f"blessed tree {blessed} has no files to compare against"]


def test_compare_accepts_npz_within_tolerance(tmp_path: Path) -> None:
    blessed = tmp_path / "blessed"
    actual = tmp_path / "actual"
    blessed.mkdir()
    actual.mkdir()
    np.savez(blessed / "camera.npz", matrix=np.array([1.0], dtype=np.float32))
    np.savez(actual / "camera.npz", matrix=np.array([1.001], dtype=np.float32))

    problems = load_verify_tolerant().compare(blessed, actual, 0.01)

    assert problems == []
