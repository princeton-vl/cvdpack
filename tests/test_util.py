from pathlib import Path

from cvdpack import main, util

STATIC_TYPE = {
    "original_path_template": "{scene}/object-data.npz",
    "packed_path_template": "{scene}/object-data.npz",
}


def test_match_template_paths_static_file(tmp_path: Path) -> None:
    path = tmp_path / "scene" / "object-data.npz"
    path.parent.mkdir()
    path.touch()

    assert list(util.match_template_paths(path)) == [({}, path)]


def test_match_template_paths_missing_static_file_returns_no_matches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene" / "object-data.npz"

    assert list(util.match_template_paths(path)) == []


def test_pack_dataset_static_scene_file(tmp_path: Path) -> None:
    source = tmp_path / "scene" / "object-data.npz"
    source.parent.mkdir()
    source.write_bytes(b"object data")
    output_folder = tmp_path / "packed"
    config = {"metadata": {}, "data_types": {"object-data": STATIC_TYPE}}

    main.pack_dataset(
        input_folder=tmp_path,
        output_folder=output_folder,
        steps=None,
        config=config,
        parallel_mode="multiprocess",
        slurm_args=None,
        n_workers=None,
        tmp_folder=tmp_path / "tmp",
        subset={"scene": ["scene"]},
        lazy=False,
    )

    assert (output_folder / "scene" / "object-data.npz").read_bytes() == b"object data"
