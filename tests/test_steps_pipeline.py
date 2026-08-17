import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cvdpack import main
from cvdpack.pack_timeseries import unpack_video
from cvdpack.util import load_any_image, match_template_paths

N_FRAMES = 3
SHAPE = (16, 24)
DEPTH_RANGE = (0.0, 10.0)
DEPTH_ATOL = 2 * (DEPTH_RANGE[1] - DEPTH_RANGE[0]) / (2**16 - 2)

DEPTH_PACKING = {
    "method": "LINEAR",
    "min_orig_val": DEPTH_RANGE[0],
    "max_orig_val": DEPTH_RANGE[1],
    "from_dtype": "float32",
    "to_dtype": "uint16",
    "out_of_bounds_method": "error",
}

DATA_TYPES = {
    "depth": {
        "original_path_template": "{scene}/depth/{frame:06d}.npy",
        "packed_path_template": "{scene}/depth.mkv",
        "packing": DEPTH_PACKING,
    },
    "pose": {
        "original_path_template": "{scene}/pose.txt",
        "packed_path_template": "{scene}/pose.npy",
    },
    "meta": {
        "original_path_template": "{scene}/meta.json",
        "packed_path_template": "{scene}/meta.json",
    },
}

CONFIG = {"metadata": {}, "data_types": DATA_TYPES}


def _write_dataset(src: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    scene = src / "P000"
    (scene / "depth").mkdir(parents=True)
    depth = rng.uniform(0.5, 9.5, (N_FRAMES, *SHAPE)).astype(np.float32)
    for frame in range(N_FRAMES):
        np.save(scene / "depth" / f"{frame:06d}.npy", depth[frame])
    pose = rng.standard_normal((N_FRAMES, 7))
    np.savetxt(scene / "pose.txt", pose)
    (scene / "meta.json").write_text(json.dumps({"fps": 24}))
    return {"depth": depth, "pose": pose}


def _run_cvdpack(
    mp: pytest.MonkeyPatch,
    action: str,
    inp: Path,
    out: Path,
    tmp: Path,
    config: Path | None = None,
    steps: list[str] | None = None,
) -> None:
    argv = ["cvdpack", action, "--input", str(inp), "--output", str(out)]
    argv += ["--tmp_folder", str(tmp)]
    if config is not None:
        argv += ["--config", str(config)]
    if steps is not None:
        argv += ["--steps", *steps]
    mp.setattr(sys, "argv", argv)
    main.main()


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("steps_pipeline")
    src = root / "src"
    src.mkdir()
    originals = _write_dataset(src)

    config_path = root / "config.json"
    config_path.write_text(json.dumps(CONFIG))

    names = ["out_a", "out_b", "mid_pack", "mid_unpack", "unpacked_a", "unpacked_b"]
    paths = {name: root / name for name in names}
    tmp = root / "tmp"

    a, b = paths["out_a"], paths["out_b"]
    ua, ub = paths["unpacked_a"], paths["unpacked_b"]
    mp_, mu = paths["mid_pack"], paths["mid_unpack"]
    with pytest.MonkeyPatch.context() as mp:
        _run_cvdpack(mp, "pack", src, a, tmp, config=config_path)
        _run_cvdpack(mp, "pack", src, mp_, tmp, config=config_path, steps=["quantize"])
        _run_cvdpack(mp, "pack", mp_, b, tmp, steps=["pack_video"])
        _run_cvdpack(mp, "unpack", a, ua, tmp)
        _run_cvdpack(mp, "unpack", b, mu, tmp, steps=["unpack_video"])
        _run_cvdpack(mp, "unpack", mu, ub, tmp, steps=["unquantize"])

    return {"src": src, "originals": originals, **paths}


def _rel_files(folder: Path) -> set[Path]:
    files = (p for p in folder.rglob("*") if p.is_file())
    return {p.relative_to(folder) for p in files} - {Path("cvdpack.json")}


def _decode_mkv(video_path: Path, out_dir: Path) -> list[np.ndarray]:
    template = out_dir / "{frame:06d}.png"
    unpack_video(video_path, template, tmp_folder=out_dir / "tmp")
    matched = sorted(match_template_paths(template), key=lambda x: x[0]["frame"])
    return [load_any_image(path) for _, path in matched]


def _load_depth(folder: Path) -> np.ndarray:
    frames = [folder / "P000" / "depth" / f"{f:06d}.npy" for f in range(N_FRAMES)]
    return np.stack([np.load(path) for path in frames])


def test_packed_file_sets_match(pipelines: dict) -> None:
    expected = {
        Path("P000/depth.mkv"),
        Path("P000/pose.npy"),
        Path("P000/meta.json"),
    }
    assert _rel_files(pipelines["out_a"]) == expected
    assert _rel_files(pipelines["out_b"]) == expected


def test_intermediate_pack_holds_quantized_pngs(pipelines: dict) -> None:
    expected = {Path(f"P000/depth/{f:06d}.png") for f in range(N_FRAMES)}
    expected |= {Path("P000/pose.npy"), Path("P000/meta.json")}
    assert _rel_files(pipelines["mid_pack"]) == expected


def test_intermediate_unpack_holds_pngs(pipelines: dict) -> None:
    expected = {Path(f"P000/depth/{f:06d}.png") for f in range(N_FRAMES)}
    expected |= {Path("P000/pose.npy"), Path("P000/meta.json")}
    assert _rel_files(pipelines["mid_unpack"]) == expected


def test_deterministic_packed_files_byte_equal(pipelines: dict) -> None:
    for rel in [Path("P000/pose.npy"), Path("P000/meta.json")]:
        packed_a = (pipelines["out_a"] / rel).read_bytes()
        packed_b = (pipelines["out_b"] / rel).read_bytes()
        assert packed_a == packed_b, rel


def test_packed_videos_decode_identically(pipelines: dict, tmp_path: Path) -> None:
    frames_a = _decode_mkv(pipelines["out_a"] / "P000/depth.mkv", tmp_path / "a")
    frames_b = _decode_mkv(pipelines["out_b"] / "P000/depth.mkv", tmp_path / "b")
    assert len(frames_a) == N_FRAMES
    assert len(frames_b) == N_FRAMES
    for frame_a, frame_b in zip(frames_a, frames_b):
        np.testing.assert_array_equal(frame_a, frame_b)


def test_unpacked_file_sets_match_original(pipelines: dict) -> None:
    expected = _rel_files(pipelines["src"])
    assert _rel_files(pipelines["unpacked_a"]) == expected
    assert _rel_files(pipelines["unpacked_b"]) == expected


def test_unpacked_depth_matches(pipelines: dict) -> None:
    depth_a = _load_depth(pipelines["unpacked_a"])
    depth_b = _load_depth(pipelines["unpacked_b"])
    np.testing.assert_array_equal(depth_a, depth_b)
    original = pipelines["originals"]["depth"]
    np.testing.assert_allclose(depth_a, original, rtol=0, atol=DEPTH_ATOL)


def test_unpacked_pose_matches(pipelines: dict) -> None:
    pose_a = np.loadtxt(pipelines["unpacked_a"] / "P000" / "pose.txt")
    pose_b = np.loadtxt(pipelines["unpacked_b"] / "P000" / "pose.txt")
    np.testing.assert_array_equal(pose_a, pose_b)
    np.testing.assert_array_equal(pose_a, pipelines["originals"]["pose"])


def test_unpacked_passthrough_matches(pipelines: dict) -> None:
    original = (pipelines["src"] / "P000" / "meta.json").read_bytes()
    assert (pipelines["unpacked_a"] / "P000" / "meta.json").read_bytes() == original
    assert (pipelines["unpacked_b"] / "P000" / "meta.json").read_bytes() == original


def test_combined_steps_in_one_call_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_dataset(src)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(CONFIG))

    with pytest.raises(ValueError, match="Unhandled steps"):
        _run_cvdpack(
            monkeypatch,
            "pack",
            src,
            tmp_path / "out",
            tmp_path / "tmp",
            config=config_path,
            steps=["quantize", "pack_video"],
        )
