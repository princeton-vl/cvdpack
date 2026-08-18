from pathlib import Path

from cvdpack.main import (
    Job,
    ambiguous_job_claims,
    decide_dataset_job_templates,
    find_jobs,
)
from cvdpack.util import match_template_paths

SCENE = "indoors_000000"

LITERAL = {
    "rgb": "{scene}/rgb-{cam}.mkv",
    "rgb-denoised": "{scene}/rgb-denoised-{cam}.mkv",
}
FIELD = {
    "rgb": "{scene}/{gt_type}-{cam}.mkv",
    "rgb-denoised": "{scene}/{gt_type}-{cam}.mkv",
}
DISTINCT = {
    "rgb": "{scene}/rgb-{cam}.mkv",
    "depth": "{scene}/depth-{cam}.mkv",
}


def make_tree(root: Path, names: list[str]) -> None:
    scene = root / SCENE
    scene.mkdir(parents=True)
    for name in names:
        (scene / name).write_bytes(b"")


def collect_unpack_jobs(root: Path, packed: dict[str, str]) -> list[Job]:
    jobs = []
    for gt_type, template in packed.items():
        conf = {
            "original_path_template": "{scene}/{cam}/" + gt_type + "_{frame:04d}.png",
            "packed_path_template": template,
        }
        search, out = decide_dataset_job_templates(
            root, root / "out", conf, None, mode="unpack"
        )
        found = find_jobs(
            input_template=search,
            output_template=out,
            gt_type=gt_type,
            subset=None,
            match_video_folder=True,
            missing_gt="silent",
        )
        jobs += [
            Job(inp, outp, gt_type, root / "tmp", conf, 1, None) for inp, outp in found
        ]
    return jobs


def test_scalar_subset_values_are_normalized(tmp_path: Path) -> None:
    make_tree(tmp_path, ["rgb-CameraLeft.mkv"])
    found = find_jobs(
        input_template=tmp_path / "{scene}/rgb-{cam}.mkv",
        output_template=tmp_path / "out" / "{scene}/rgb-{cam}.mkv",
        gt_type="rgb",
        subset={"scene": SCENE},
        missing_gt="silent",
    )
    assert len(found) == 1


def test_repeated_numeric_field_across_the_folder_split(tmp_path: Path) -> None:
    scene = tmp_path / "0001"
    scene.mkdir()
    (scene / "0001_left_000000.png").write_bytes(b"")
    found = find_jobs(
        input_template=tmp_path / "{scene:04d}" / "{scene}_{view}_{frame:06d}.png",
        output_template=tmp_path / "out" / "{scene}" / "{view}_{frame:06d}.png",
        gt_type="rgb",
        match_video_folder=True,
        missing_gt="silent",
    )
    assert len(found) == 1
    assert "0001" in str(found[0][1])


def test_plain_repeated_fields_must_match_exactly(tmp_path: Path) -> None:
    scene = tmp_path / "0001"
    scene.mkdir()
    (scene / "1_left_000000.png").write_bytes(b"")
    found = find_jobs(
        input_template=tmp_path / "{scene}" / "{scene}_{view}_{frame:06d}.png",
        output_template=tmp_path / "out" / "{scene}" / "{view}_{frame:06d}.png",
        gt_type="rgb",
        match_video_folder=True,
        missing_gt="silent",
    )
    assert found == []


def test_repeated_field_keeps_the_parent_spelling(tmp_path: Path) -> None:
    scene = tmp_path / "0001"
    scene.mkdir()
    (scene / "0001_left_000000.png").write_bytes(b"")
    found = find_jobs(
        input_template=tmp_path / "{scene}" / "{scene:04d}_{view}_{frame:06d}.png",
        output_template=tmp_path / "out" / "{scene}" / "{view}_{frame:06d}.png",
        gt_type="rgb",
        match_video_folder=True,
        missing_gt="silent",
    )
    assert len(found) == 1
    assert "/0001/" in str(found[0][1])


def test_a_broad_literal_template_is_reported_as_ambiguous(tmp_path: Path) -> None:
    make_tree(tmp_path, ["rgb-CameraLeft.mkv", "rgb-denoised-CameraLeft.mkv"])

    jobs = collect_unpack_jobs(tmp_path, LITERAL)
    ambiguous = ambiguous_job_claims(jobs)

    assert list(ambiguous) == [tmp_path / SCENE / "rgb-denoised-CameraLeft.mkv"]
    assert sorted(*ambiguous.values()) == ["rgb", "rgb-denoised"]


def test_a_gt_type_field_is_not_ambiguous(tmp_path: Path) -> None:
    make_tree(tmp_path, ["rgb-CameraLeft.mkv", "rgb-denoised-CameraLeft.mkv"])

    jobs = collect_unpack_jobs(tmp_path, FIELD)

    assert ambiguous_job_claims(jobs) == {}
    assert {job.input_path.name: job.gt_type for job in jobs} == {
        "rgb-CameraLeft.mkv": "rgb",
        "rgb-denoised-CameraLeft.mkv": "rgb-denoised",
    }


def test_distinct_literal_templates_are_not_ambiguous(tmp_path: Path) -> None:
    make_tree(tmp_path, ["rgb-CameraLeft.mkv", "depth-CameraLeft.mkv"])

    jobs = collect_unpack_jobs(tmp_path, DISTINCT)

    assert ambiguous_job_claims(jobs) == {}
    assert len(jobs) == 2


def test_no_jobs_are_never_ambiguous() -> None:
    assert ambiguous_job_claims([]) == {}


def test_a_repeated_field_must_match_its_first_occurrence(tmp_path: Path) -> None:
    flow = tmp_path / SCENE / "flow"
    flow.mkdir(parents=True)
    (flow / "000000_000001_flow.npy").write_bytes(b"")
    (flow / "000000_000001_mask.npy").write_bytes(b"")

    template = "{scene}/{gt_type}/{frame:06d}_{framenext:06d}_{gt_type}.npy"
    matches = [
        (info["gt_type"], path.name)
        for info, path in match_template_paths(tmp_path / template)
    ]

    assert matches == [("flow", "000000_000001_flow.npy")]
