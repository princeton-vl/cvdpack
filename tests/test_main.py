import logging
from pathlib import Path

from cvdpack import main
from cvdpack import util


def _find_jobs(
    tmp_path: Path, gt_type: str, subset_strings: list[str]
) -> list[main.Job]:
    subset = util.parse_dictlist_strings(subset_strings)
    return main.find_jobs(
        input_template=tmp_path / "packed/{scene}" / f"{gt_type}-{{cam}}.mkv",
        output_template=tmp_path / "out/{scene}/{cam}" / f"{gt_type}.mkv",
        gt_type=gt_type,
        subset=subset,
        job_defaults={
            "gt_type": gt_type,
            "subset": subset,
            "tmp_folder": tmp_path / "tmp",
            "config": {},
            "cpus_per_worker": None,
            "loglevel": logging.WARNING,
        },
        missing_gt="silent",
    )


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


def test_included_in_filter_matches_any_listed_value() -> None:
    keys = {"scene": "scene-a", "cam": "CameraLeft"}
    assert util.included_in_filter(keys, {"cam": ["CameraLeft", "CameraRight"]})
    assert util.included_in_filter(keys, {"cam": "CameraLeft"})
    assert not util.included_in_filter(keys, {"cam": ["CameraRight"]})
