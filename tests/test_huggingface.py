import argparse
import json
import types
from pathlib import Path
from unittest import mock

import pytest

from cvdpack import huggingface, main

CONFIG = {
    "data_types": {
        "rgb": {"packed_path_template": "{scene}/rgb-{cam}.mkv"},
        "metadata": {"packed_path_template": "{scene}/metadata.json"},
    }
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://huggingface.co/datasets/org/dataset", ("org/dataset", "main", "")),
        (
            "https://huggingface.co/datasets/org/dataset/tree/rev/test",
            ("org/dataset", "rev", "test"),
        ),
        (
            "https://huggingface.co/datasets/org/dataset/tree/feature%2Ftest/data",
            ("org/dataset", "feature/test", "data"),
        ),
        (
            "https://huggingface.co/datasets/org/dataset/tree/main/my%20data/scene",
            ("org/dataset", "main", "my data/scene"),
        ),
    ],
)
def test_parse_input(value: str, expected: tuple[str, str, str]) -> None:
    assert huggingface.parse_input(value) == expected


def test_validate_args_keeps_huggingface_source_out_of_cli_args() -> None:
    args = argparse.Namespace(
        action="copy",
        config=None,
        hf_staging=None,
        input="https://huggingface.co/datasets/org/dataset",
        min_tmp_folder_space_mb=0,
        n_workers=None,
        parallel_mode="none",
        steps=[],
        tmp_folder=None,
    )

    original_args = vars(args).copy()
    input_path, config_path, tmp_folder, hf_source = main.validate_args(args)

    assert hf_source == ("org/dataset", "main", "")
    assert input_path is None
    assert config_path is None
    assert tmp_folder is None
    assert vars(args) == original_args


def test_download_patterns_crosses_selected_path_fields() -> None:
    patterns = huggingface.download_patterns(
        CONFIG,
        {
            "scene": ["scene-a", "scene-b"],
            "gt_type": "rgb",
            "cam": ["left", "right"],
        },
        "test/",
    )

    assert patterns == [
        "test/scene-a/rgb-left.mkv",
        "test/scene-a/rgb-right.mkv",
        "test/scene-b/rgb-left.mkv",
        "test/scene-b/rgb-right.mkv",
    ]


def test_download_patterns_formats_numeric_fields() -> None:
    config = {
        "data_types": {"rgb": {"packed_path_template": "{scene:04d}/rgb-{cam}.mkv"}}
    }

    patterns = huggingface.download_patterns(config, {"scene": "1"}, "")

    assert patterns == ["0001/rgb-*.mkv"]


@pytest.mark.parametrize(
    ("scene", "other"),
    [("run*", "runner"), ("run?", "run1"), ("run[1]", "run1")],
)
def test_download_patterns_treats_selected_glob_characters_literally(
    scene: str, other: str
) -> None:
    patterns = huggingface.download_patterns(CONFIG, {"scene": scene}, "")
    files = [f"{scene}/rgb-left.mkv", f"{other}/rgb-left.mkv"]

    assert huggingface.matching_files(files, patterns) == [files[0]]


def test_download_patterns_rejects_unknown_subset_keys() -> None:
    with pytest.raises(ValueError, match="sceen.*scene"):
        huggingface.download_patterns(CONFIG, {"sceen": "scene-a"}, "")


def test_retain_config_copies_an_external_config_into_the_destination(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "external" / "cvdpack.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(CONFIG))
    data_folder = tmp_path / "packed" / "test"

    retained = huggingface.retain_config(config_path, data_folder)

    assert retained == data_folder / "cvdpack.json"
    assert json.loads(retained.read_text()) == CONFIG


def test_retain_config_keeps_a_config_already_in_the_destination(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "cvdpack.json"
    config_path.write_text(json.dumps(CONFIG))

    assert huggingface.retain_config(config_path, tmp_path) == config_path


def test_matching_files_selects_the_requested_patterns() -> None:
    repo_files = [
        "cvdpack.json",
        "test/scene-a/rgb-left.mkv",
        "test/scene-a/depth-left.mkv",
        "test/scene-b/rgb-left.mkv",
    ]
    patterns = huggingface.download_patterns(
        CONFIG, {"scene": "scene-a", "gt_type": "rgb"}, "test/"
    )

    assert huggingface.matching_files(repo_files, patterns) == [
        "test/scene-a/rgb-left.mkv"
    ]


def test_matching_files_is_empty_for_an_unreachable_subset() -> None:
    patterns = huggingface.download_patterns(
        CONFIG, {"scene": "scene-a/scene-a-traj0", "gt_type": "rgb"}, "test/"
    )

    assert huggingface.matching_files(["test/scene-a/rgb-left.mkv"], patterns) == []


def test_matching_files_keeps_wildcards_within_one_path_component() -> None:
    patterns = huggingface.download_patterns(CONFIG, {}, "")
    files = ["scene/rgb-left.mkv", "nested/scene/rgb-left.mkv"]

    assert huggingface.matching_files(files, patterns) == [files[0]]


def test_select_config_file_prefers_the_repository_root() -> None:
    repo_files = ["cvdpack.json", "test/cvdpack.json", "test/scene-a/rgb-left.mkv"]

    assert huggingface.select_config_file(repo_files, "test/") == "cvdpack.json"


def test_select_config_file_falls_back_to_the_subpath() -> None:
    repo_files = ["test/cvdpack.json", "test/scene-a/rgb-left.mkv"]

    assert huggingface.select_config_file(repo_files, "test/") == "test/cvdpack.json"


def test_select_config_file_names_both_locations_when_missing() -> None:
    with pytest.raises(FileNotFoundError, match="cvdpack.json.*test/cvdpack.json"):
        huggingface.select_config_file(["test/scene-a/rgb-left.mkv"], "test/")


def test_download_patterns_ignores_unused_subset_fields() -> None:
    patterns = huggingface.download_patterns(
        CONFIG,
        {"scene": "scene-a", "gt_type": "metadata", "cam": "left"},
        "test/",
    )

    assert patterns == ["test/scene-a/metadata.json"]


def test_download_input_checks_compatibility_before_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "cvdpack.json"
    config = {**CONFIG, "metadata": {"compatibility_version": 999}}
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(return_value=str(config_path))
    hub.list_repo_files = mock.Mock(return_value=["cvdpack.json", "scene/rgb.mkv"])
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    with pytest.raises(ValueError, match="compatibility version 999"):
        huggingface.download_input(
            ("org/dataset", "main", ""), tmp_path / "dest", None, None, 1
        )

    hub.snapshot_download.assert_not_called()


def test_download_input_matches_root_config_before_scoping_subpath(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "cvdpack.json"
    config = {**CONFIG, "metadata": {"compatibility_version": 1}}
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(return_value=str(config_path))
    hub.list_repo_files = mock.Mock(
        return_value=["cvdpack.json", "test/rgb-left.mkv", "other/rgb-left.mkv"]
    )
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    data_root, _, _ = huggingface.download_input(
        ("org/dataset", "main", "test"), tmp_path / "dest", None, None, 1
    )

    assert hub.snapshot_download.call_args.kwargs["allow_patterns"] == [
        "test/rgb-left.mkv"
    ]
    assert data_root == tmp_path / "dest"


def test_download_input_prefixes_templates_for_an_external_config(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "external.json"
    config = {**CONFIG, "metadata": {"compatibility_version": 1}}
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock()
    hub.list_repo_files = mock.Mock(
        return_value=["test/scene-a/rgb-left.mkv", "other/scene-a/rgb-left.mkv"]
    )
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    data_root, retained, _ = huggingface.download_input(
        ("org/dataset", "main", "test"), tmp_path / "dest", config_path, None, 1
    )

    assert hub.snapshot_download.call_args.kwargs["allow_patterns"] == [
        "test/scene-a/rgb-left.mkv"
    ]
    assert data_root == tmp_path / "dest" / "test"
    assert retained == data_root / "cvdpack.json"


def test_download_input_accepts_numeric_template_fields(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "metadata": {"compatibility_version": 1},
        "data_types": {"rgb": {"packed_path_template": "{scene:04d}/rgb.mkv"}},
    }
    config_path = tmp_path / "cvdpack.json"
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(return_value=str(config_path))
    hub.list_repo_files = mock.Mock(return_value=["cvdpack.json", "0001/rgb.mkv"])
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    huggingface.download_input(
        ("org/dataset", "main", ""), tmp_path / "dest", None, None, 1
    )

    assert hub.snapshot_download.call_args.kwargs["allow_patterns"] == ["0001/rgb.mkv"]


def test_download_input_accepts_separator_valued_subsets(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "metadata": {"compatibility_version": 1},
        "data_types": {"camera": {"packed_path_template": "{scene}/camera.npz"}},
    }
    config_path = tmp_path / "cvdpack.json"
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(return_value=str(config_path))
    hub.list_repo_files = mock.Mock(
        return_value=[
            "cvdpack.json",
            "id/id_traj0/camera.npz",
            "other/other_traj0/camera.npz",
        ]
    )
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    huggingface.download_input(
        ("org/dataset", "main", ""),
        tmp_path / "dest",
        None,
        {"scene": ["id/id_traj0"]},
        1,
    )

    assert hub.snapshot_download.call_args.kwargs["allow_patterns"] == [
        "id/id_traj0/camera.npz"
    ]


def test_download_input_per_job_stages_placeholders(
    tmp_path: Path, monkeypatch
) -> None:
    config = {**CONFIG, "metadata": {"compatibility_version": 1}}
    config_path = tmp_path / "cvdpack.json"
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(return_value=str(config_path))
    hub.list_repo_files = mock.Mock(
        return_value=["cvdpack.json", "scene-a/rgb-left.mkv"]
    )
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    data_root, _, remote = huggingface.download_input(
        ("org/dataset", "main", ""),
        tmp_path / "dest",
        None,
        None,
        1,
        staging="per_job",
    )

    hub.snapshot_download.assert_not_called()
    placeholder = tmp_path / "dest" / "scene-a" / "rgb-left.mkv"
    assert placeholder.exists()
    assert placeholder.stat().st_size == 0
    assert remote == {
        "repo_id": "org/dataset",
        "revision": "main",
        "dest": str(tmp_path / "dest"),
    }


def test_fetch_job_input_downloads_the_repo_relative_file(
    tmp_path: Path, monkeypatch
) -> None:
    placeholder = tmp_path / "scene-a" / "rgb-left.mkv"
    placeholder.parent.mkdir(parents=True)
    placeholder.touch()

    def download(*args, **kwargs) -> None:
        assert not placeholder.exists()

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(side_effect=download)
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)
    remote = {"repo_id": "org/dataset", "revision": "rev", "dest": str(tmp_path)}

    huggingface.fetch_job_input(remote, placeholder)

    hub.hf_hub_download.assert_called_once_with(
        "org/dataset",
        "scene-a/rgb-left.mkv",
        repo_type="dataset",
        revision="rev",
        local_dir=Path(tmp_path),
    )


def test_download_input_rejects_repeated_field_mismatches(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "metadata": {"compatibility_version": 1},
        "data_types": {
            "rgb": {"packed_path_template": "{scene}/frames/{scene}_rgb.mkv"}
        },
    }
    config_path = tmp_path / "cvdpack.json"
    config_path.write_text(json.dumps(config))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(return_value=str(config_path))
    hub.list_repo_files = mock.Mock(
        return_value=[
            "cvdpack.json",
            "sceneA/frames/sceneA_rgb.mkv",
            "sceneA/frames/sceneB_rgb.mkv",
        ]
    )
    hub.snapshot_download = mock.Mock()
    monkeypatch.setattr(huggingface, "huggingface_hub", hub)

    huggingface.download_input(
        ("org/dataset", "main", ""), tmp_path / "dest", None, None, 1
    )

    assert hub.snapshot_download.call_args.kwargs["allow_patterns"] == [
        "sceneA/frames/sceneA_rgb.mkv"
    ]
