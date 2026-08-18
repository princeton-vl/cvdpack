import itertools
import json
import logging
from pathlib import Path

import pytest

from cvdpack import main, util


def _find_jobs(
    tmp_path: Path, gt_type: str, subset_strings: list[str]
) -> list[tuple[Path, Path]]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "packed/{scene}" / f"{gt_type}-{{cam}}.mkv",
        output_template=tmp_path / "out/{scene}/{cam}" / f"{gt_type}.mkv",
        gt_type=gt_type,
        subset=subset,
        missing_gt="silent",
    )


def _find_traj_jobs(
    tmp_path: Path, subset_strings: list[str] | None
) -> list[tuple[Path, Path]]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "packed/{scene}/{scene}_traj{traj}/rgb-{cam}.mkv",
        output_template=tmp_path / "out/{scene}/{traj}/{cam}/rgb.mkv",
        gt_type="rgb",
        subset=subset,
        missing_gt="silent",
    )


def _find_view_jobs(
    tmp_path: Path, subset_strings: list[str]
) -> list[tuple[Path, Path]]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "frames/{scene}/{frame:06d}_{view}.png",
        output_template=tmp_path / "packed/{scene}/rgb-{view}.mkv",
        gt_type="rgb",
        subset=subset,
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
        inp.name
        for gt_type in ["rgb", "depth", "surface-normal"]
        for inp, _ in _find_jobs(tmp_path, gt_type, subset)
    ]
    assert found == ["rgb-CameraLeft.mkv", "depth-CameraLeft.mkv"]


def test_find_jobs_comma_separated_subset_stays_a_filter(tmp_path: Path) -> None:
    _stage(tmp_path)

    jobs = _find_jobs(tmp_path, "rgb", ["gt_type=rgb", "cam=CameraLeft,CameraRight"])
    assert [inp.name for inp, _ in jobs] == [
        "rgb-CameraLeft.mkv",
        "rgb-CameraRight.mkv",
    ]
    assert [out.parent.name for _, out in jobs] == [
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
    assert {inp.parent.name.split("_traj")[1] for inp, _ in jobs} == {"0"}


def test_find_jobs_comma_separated_numeric_subset(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["traj=0,2"])
    assert len(jobs) == 8
    assert {inp.parent.name.split("_traj")[1] for inp, _ in jobs} == {"0", "2"}
    assert {out.parent.parent.name for _, out in jobs} == {"0", "2"}


def test_find_jobs_zero_padded_subset(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["scene=0001"])
    assert len(jobs) == 8
    assert {inp.parts[-3] for inp, _ in jobs} == {"0001"}
    assert {out.parts[-4] for _, out in jobs} == {"0001"}


def test_find_jobs_zero_padded_subset_as_multivalued_filter(tmp_path: Path) -> None:
    _stage_trajs(tmp_path)

    jobs = _find_traj_jobs(tmp_path, ["scene=0001,scene-a"])
    assert len(jobs) == 16
    assert {inp.parts[-3] for inp, _ in jobs} == {"0001", "scene-a"}


def test_find_jobs_keeps_zero_padded_multivalue_output_names(tmp_path: Path) -> None:
    for scene in ["0001", "1", "scene-a"]:
        path = tmp_path / "packed" / scene / f"{scene}_traj0/rgb-CameraLeft.mkv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    jobs = _find_traj_jobs(tmp_path, ["scene=0001,scene-a"])

    assert len(jobs) == 2
    assert {inp.parts[-3] for inp, _ in jobs} == {"0001", "scene-a"}
    assert {out.parts[-4] for _, out in jobs} == {"0001", "scene-a"}


def test_find_jobs_filters_discovered_filename_groups(tmp_path: Path) -> None:
    _stage_views(tmp_path)

    jobs = _find_view_jobs(tmp_path, ["view=left,right"])

    assert len(jobs) == 2
    assert {inp.name for inp, _ in jobs} == {
        "{frame:06d}_left.png",
        "{frame:06d}_right.png",
    }
    assert {out.name for _, out in jobs} == {"rgb-left.mkv", "rgb-right.mkv"}


def test_find_jobs_filename_filter_selects_one_group(tmp_path: Path) -> None:
    _stage_views(tmp_path)

    jobs = _find_view_jobs(tmp_path, ["view=left"])

    assert [inp.name for inp, _ in jobs] == ["{frame:06d}_left.png"]
    assert [out.name for _, out in jobs] == ["rgb-left.mkv"]


def test_find_jobs_discovers_filename_groups_without_subset(tmp_path: Path) -> None:
    _stage_views(tmp_path)

    jobs = _find_view_jobs(tmp_path, [])

    assert {inp.name for inp, _ in jobs} == {
        "{frame:06d}_left.png",
        "{frame:06d}_right.png",
    }
    assert {out.name for _, out in jobs} == {"rgb-left.mkv", "rgb-right.mkv"}


def test_find_jobs_skips_a_filename_disagreeing_with_its_parent(
    tmp_path: Path,
) -> None:
    for name in ["scene-a_right_000001.png", "scene-b_left_000001.png"]:
        path = tmp_path / "frames/scene-a" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    jobs = main.find_jobs(
        input_template=tmp_path / "frames/{scene}/{scene}_{view}_{frame:06d}.png",
        output_template=tmp_path / "out/{scene}/{view}.mkv",
        gt_type="rgb",
        subset=None,
        match_video_folder=True,
        missing_gt="silent",
    )

    # scene-b_left lives under scene-a, so it matches no consistent {scene}
    assert [out.name for _, out in jobs] == ["right.mkv"]
    assert [out.parent.name for _, out in jobs] == ["scene-a"]


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
        missing_gt="silent",
    )
    assert [inp.name for inp, _ in jobs] == ["0001.png"]
    assert [out.name for _, out in jobs] == ["0001.png"]


@pytest.mark.parametrize("frame_subset", ["frame=0,1", "frame=1", "framenext=1"])
def test_find_jobs_frame_subset_on_a_sequence_is_rejected(
    tmp_path: Path, frame_subset: str
) -> None:
    _stage_views(tmp_path)

    with pytest.raises(ValueError, match="cannot select frames"):
        _find_view_jobs(tmp_path, [frame_subset])


def test_find_jobs_frame_subset_still_works_per_frame(tmp_path: Path) -> None:
    for frame in range(4):
        path = tmp_path / "packed/scene-a/CameraLeft" / f"{frame:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    subset = util.parse_dictlist_strings(["frame=1,3"])
    jobs = main.find_jobs(
        input_template=tmp_path / "packed/{scene}/{cam}/{frame:04d}.png",
        output_template=tmp_path / "out/{scene}/{cam}/{frame:04d}.png",
        gt_type="rgb",
        subset=subset,
        missing_gt="silent",
    )
    assert sorted(inp.name for inp, _ in jobs) == ["0001.png", "0003.png"]


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


def _job(input_path: Path, output_path: Path) -> main.Job:
    return main.Job(
        input_path=input_path,
        output_path=output_path,
        gt_type="rgb",
        tmp_folder=None,
        config={},
        cpus_per_worker=None,
        loglevel=logging.WARNING,
    )


def test_validate_jobs_rejects_output_collision(tmp_path: Path) -> None:
    output_path = tmp_path / "output.mkv"
    jobs = [_job(tmp_path / name, output_path) for name in ["left.png", "right.png"]]

    with pytest.raises(ValueError, match="both write"):
        main.validate_jobs(jobs)


def test_validate_jobs_allows_runtime_output_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "output/{resample}_{cam_rig}_{subcam}.png"
    job = _job(tmp_path / "input.png", output_path)

    main.validate_jobs([job])


def test_process_video_job_uses_disposable_tmp_folder(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("1.0 2.0")
    tmp_folder = tmp_path / "tmp"
    job = main.Job(
        input_path=input_path,
        output_path=tmp_path / "deep/output.txt",
        gt_type="metadata",
        tmp_folder=tmp_folder,
        config={},
        cpus_per_worker=None,
        loglevel=logging.WARNING,
    )

    main.process_video_job(job)
    main.process_video_job(job)

    assert job.output_path.read_text() == "1.0 2.0"
    assert list(tmp_folder.iterdir()) == []
