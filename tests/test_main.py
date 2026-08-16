import itertools
import json
import logging
from pathlib import Path

import pytest

from cvdpack import main, util


def _job_defaults(tmp_path: Path, gt_type: str, subset: dict | None) -> dict:
    return {
        "gt_type": gt_type,
        "subset": subset,
        "tmp_folder": tmp_path / "tmp",
        "config": {},
        "cpus_per_worker": None,
        "loglevel": logging.WARNING,
    }


def _find_jobs(
    tmp_path: Path, gt_type: str, subset_strings: list[str]
) -> list[main.Job]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "packed/{scene}" / f"{gt_type}-{{cam}}.mkv",
        output_template=tmp_path / "out/{scene}/{cam}" / f"{gt_type}.mkv",
        gt_type=gt_type,
        subset=subset,
        job_defaults=_job_defaults(tmp_path, gt_type, subset),
        missing_gt="silent",
    )


def _find_traj_jobs(tmp_path: Path, subset_strings: list[str] | None) -> list[main.Job]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "packed/{scene}/{scene}_traj{traj}/rgb-{cam}.mkv",
        output_template=tmp_path / "out/{scene}/{traj}/{cam}/rgb.mkv",
        gt_type="rgb",
        subset=subset,
        job_defaults=_job_defaults(tmp_path, "rgb", subset),
        missing_gt="silent",
    )


def _find_view_jobs(tmp_path: Path, subset_strings: list[str]) -> list[main.Job]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "frames/{scene}/{frame:06d}_{view}.png",
        output_template=tmp_path / "packed/{scene}/rgb-{view}.mkv",
        gt_type="rgb",
        subset=subset,
        job_defaults=_job_defaults(tmp_path, "rgb", subset),
        match_video_folder=True,
        missing_gt="silent",
    )


def _stage_trajs(tmp_path: Path) -> None:
    scenes = ["scene-a", "0001"]
    cams = ["CameraLeft", "CameraRight"]
    for scene, traj, cam in itertools.product(scenes, range(4), cams):
        path = tmp_path / "packed" / scene / f"{scene}_traj{traj}" / f"rgb-{cam}.mkv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _stage_views(tmp_path: Path) -> None:
    for frame, view in itertools.product(range(2), ["left", "right"]):
        path = tmp_path / "frames/scene-a" / f"{frame:06d}_{view}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _stage(tmp_path: Path) -> None:
    for gt_type in ["rgb", "depth", "surface-normal"]:
        for cam in ["CameraLeft", "CameraRight"]:
            path = tmp_path / "packed/scene-a" / f"{gt_type}-{cam}.mkv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")


def test_find_jobs_single_gt_type_subset(tmp_path: Path) -> None:
    _stage(tmp_path)

    assert len(_find_jobs(tmp_path, "rgb", ["gt_type=rgb", "cam=CameraLeft"])) == 1
    assert _find_jobs(tmp_path, "depth", ["gt_type=rgb", "cam=CameraLeft"]) == []


def test_find_jobs_comma_separated_gt_type_subset(tmp_path: Path) -> None:
    _stage(tmp_path)

    subset = ["gt_type=rgb,depth", "cam=CameraLeft"]
    found = [
        job.input_path.name
        for gt_type in ["rgb", "depth", "surface-normal"]
        for job in _find_jobs(tmp_path, gt_type, subset)
    ]
    assert found == ["rgb-CameraLeft.mkv", "depth-CameraLeft.mkv"]


def test_find_jobs_comma_separated_subset_stays_a_filter(tmp_path: Path) -> None:
    _stage(tmp_path)

    jobs = _find_jobs(tmp_path, "rgb", ["gt_type=rgb", "cam=CameraLeft,CameraRight"])
    assert [job.input_path.name for job in jobs] == [
        "rgb-CameraLeft.mkv",
        "rgb-CameraRight.mkv",
    ]
    assert [job.output_path.parent.name for job in jobs] == [
        "CameraLeft",
        "CameraRight",
    ]


def test_match_template_paths_coerces_only_numeric_spec_fields(tmp_path: Path) -> None:
    path = tmp_path / "0001" / "0007.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")

    matched = list(util.match_template_paths(tmp_path / "{scene}/{frame:04d}.png"))
    assert [info for info, _ in matched] == [{"scene": "0001", "frame": 7}]


def test_parse_dictlist_strings_always_yields_lists() -> None:
    parsed = util.parse_dictlist_strings(["cam=CameraLeft", "gt_type=rgb,depth"])
    assert parsed == {"cam": ["CameraLeft"], "gt_type": ["rgb", "depth"]}


def test_included_in_filter_matches_any_listed_value() -> None:
    keys = {"scene": "scene-a", "cam": "CameraLeft"}
    assert util.included_in_filter(keys, {"cam": ["CameraLeft", "CameraRight"]})
    assert util.included_in_filter(keys, {"cam": ["CameraLeft"]})
    assert not util.included_in_filter(keys, {"cam": ["CameraRight"]})


def test_included_in_filter_keeps_zero_padded_values_distinct() -> None:
    assert util.included_in_filter({"traj": "0"}, {"traj": ["0"]})
    assert util.included_in_filter({"traj": "0"}, {"traj": ["0", "2"]})
    assert util.included_in_filter({"traj": "2"}, {"traj": ["0", "2"]})
    assert not util.included_in_filter({"traj": "1"}, {"traj": ["0", "2"]})
    assert not util.included_in_filter({"scene": "1"}, {"scene": ["0001"]})


def test_find_jobs_no_subset_finds_every_traj(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    assert len(_find_traj_jobs(tmp_path, None)) == 16


def test_find_jobs_single_numeric_subset(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["traj=0"])
    assert len(jobs) == 4
    assert {job.input_path.parent.name.split("_traj")[1] for job in jobs} == {"0"}


def test_find_jobs_comma_separated_numeric_subset(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["traj=0,2"])
    assert len(jobs) == 8
    assert {job.input_path.parent.name.split("_traj")[1] for job in jobs} == {"0", "2"}
    assert {job.output_path.parent.parent.name for job in jobs} == {"0", "2"}


def test_find_jobs_zero_padded_subset(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["scene=0001"])
    assert len(jobs) == 8
    assert {job.input_path.parts[-3] for job in jobs} == {"0001"}
    assert {job.output_path.parts[-4] for job in jobs} == {"0001"}


def test_find_jobs_zero_padded_subset_as_multivalued_filter(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["scene=0001,scene-a"])
    assert len(jobs) == 16
    assert {job.input_path.parts[-3] for job in jobs} == {"0001", "scene-a"}


def test_find_jobs_keeps_zero_padded_multivalue_output_names(tmp_path: Path) -> None:
    for scene in ["0001", "1", "scene-a"]:
        path = tmp_path / "packed" / scene / f"{scene}_traj0/rgb-CameraLeft.mkv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    jobs = _find_traj_jobs(tmp_path, ["scene=0001,scene-a"])

    assert len(jobs) == 2
    assert {job.input_path.parts[-3] for job in jobs} == {"0001", "scene-a"}
    assert {job.output_path.parts[-4] for job in jobs} == {"0001", "scene-a"}


def test_find_jobs_expands_filename_multivalue_filters(tmp_path: Path) -> None:
    _stage_views(tmp_path)

    jobs = _find_view_jobs(tmp_path, ["view=left,right"])

    assert len(jobs) == 2
    assert {job.input_path.name for job in jobs} == {
        "{frame:06d}_left.png",
        "{frame:06d}_right.png",
    }
    assert {job.output_path.name for job in jobs} == {"rgb-left.mkv", "rgb-right.mkv"}


def test_find_jobs_numeric_format_spec_subset(tmp_path: Path) -> None:
    for frame in range(4):
        path = tmp_path / "packed/scene-a/CameraLeft" / f"{frame:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    subset = util.parse_dictlist_strings(["frame=1"])
    jobs = main.find_jobs(
        input_template=tmp_path / "packed/{scene}/{cam}/{frame:04d}.png",
        output_template=tmp_path / "out/{scene}/{cam}/{frame:04d}.png",
        gt_type="rgb",
        subset=subset,
        job_defaults=_job_defaults(tmp_path, "rgb", subset),
        missing_gt="silent",
    )
    assert [job.input_path.name for job in jobs] == ["0001.png"]
    assert [job.output_path.name for job in jobs] == ["0001.png"]


def test_format_template_numeric_spec_accepts_string_value() -> None:
    template = "{scene}/{frame:04d}.png"
    assert util.format_template(template, {"scene": "s", "frame": "1"}) == "s/0001.png"
    assert util.format_template(template, {"scene": "s", "frame": 1}) == "s/0001.png"


def test_format_template_keeps_zero_padded_string_unspecced() -> None:
    assert util.format_template("{scene}/x.png", {"scene": "0001"}) == "0001/x.png"


def _copy_argv(input_folder: Path, output_folder: Path) -> list[str]:
    return [
        "cvdpack",
        "copy",
        "--input",
        str(input_folder / "{frame:06d}.png"),
        "--output",
        str(output_folder / "{frame:06d}.png"),
        # an omitted --subset is parsed as None, which currently filters out every file
        "--subset",
    ]


def _write_frames(input_folder: Path, n_frames: int) -> None:
    input_folder.mkdir()
    for frame in range(n_frames):
        (input_folder / f"{frame:06d}.png").write_bytes(b"")


def test_copy_without_config(tmp_path: Path, monkeypatch) -> None:
    input_folder = tmp_path / "input"
    _write_frames(input_folder, 3)
    assert not (input_folder / "cvdpack.json").exists()

    output_folder = tmp_path / "output"
    monkeypatch.setattr("sys.argv", _copy_argv(input_folder, output_folder))

    main.main()

    assert sorted(p.name for p in output_folder.iterdir()) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]


def test_copy_ignores_an_incompatible_config(tmp_path: Path, monkeypatch) -> None:
    input_folder = tmp_path / "input"
    _write_frames(input_folder, 2)

    config = {
        "metadata": {"compatibility_version": "999.0", "cvdpack_version": "999.0.0"}
    }
    (input_folder / "cvdpack.json").write_text(json.dumps(config))

    output_folder = tmp_path / "output"
    monkeypatch.setattr("sys.argv", _copy_argv(input_folder, output_folder))

    main.main()

    assert sorted(p.name for p in output_folder.iterdir()) == [
        "000000.png",
        "000001.png",
    ]


def test_copy_rejects_an_explicit_config(tmp_path: Path, monkeypatch) -> None:
    input_folder = tmp_path / "input"
    _write_frames(input_folder, 2)
    (input_folder / "cvdpack.json").write_text(json.dumps({"metadata": {}}))

    output_folder = tmp_path / "output"
    argv = _copy_argv(input_folder, output_folder)
    argv[argv.index("--subset") :] = [
        "--config",
        str(input_folder / "cvdpack.json"),
        "--subset",
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(ValueError, match="copy does not read a config"):
        main.main()

    assert not output_folder.exists()
