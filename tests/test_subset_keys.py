import json
import sys

import pytest

from cvdpack import main, util

CONFIG = {
    "data_types": {
        "rgb": {
            "original_path_template": "{scene}/{cam}/{frame:04d}.png",
            "packed_path_template": "{scene}/rgb-{cam}.mkv",
        },
        "metadata": {
            "original_path_template": "{scene}/metadata.json",
            "packed_path_template": "{scene}/metadata.json",
        },
    }
}


def test_template_fields():
    assert util.template_fields("{scene}/{cam}/{frame:04d}.png") == {
        "scene",
        "cam",
        "frame",
    }
    assert util.template_fields("{scene}/metadata.json") == {"scene"}
    assert util.template_fields("nothing.json") == set()


def test_config_subset_keys_unions_all_templates():
    assert util.config_subset_keys(CONFIG) == {"gt_type", "scene", "cam", "frame"}


def test_validate_subset_keys_accepts_known_keys():
    allowed = util.config_subset_keys(CONFIG)
    util.validate_subset_keys({"scene": "a", "cam": "left"}, allowed)
    util.validate_subset_keys({"gt_type": "rgb"}, allowed)


def test_validate_subset_keys_accepts_key_missing_from_one_template():
    util.validate_subset_keys(
        {"gt_type": "metadata", "cam": "left"}, util.config_subset_keys(CONFIG)
    )


def test_validate_subset_keys_rejects_unknown_key():
    with pytest.raises(ValueError, match="traj"):
        util.validate_subset_keys(
            {"scene": "a", "traj": "0"}, util.config_subset_keys(CONFIG)
        )


def test_validate_subset_keys_error_lists_available_keys():
    with pytest.raises(ValueError, match="scene"):
        util.validate_subset_keys({"traj": "0"}, util.config_subset_keys(CONFIG))


def test_validate_subset_keys_noop_without_subset():
    util.validate_subset_keys(None, util.config_subset_keys(CONFIG))
    util.validate_subset_keys({}, util.config_subset_keys(CONFIG))


def test_template_fields_ignores_positional_field():
    assert util.template_fields("folder/{}") == set()


def write_copy_dataset(root):
    (root / "input" / "alice").mkdir(parents=True)
    (root / "input" / "alice" / "left.png").write_bytes(b"a")
    config_path = root / "cvdpack.json"
    with config_path.open("w") as f:
        json.dump(CONFIG, f)
    return config_path


def run_main(monkeypatch, action, inp, out, config_path, subset):
    argv = ["cvdpack", action]
    argv += ["--input", str(inp), "--output", str(out)]
    argv += ["--config", str(config_path), "--subset", subset]
    argv += ["--steps", "quantize"]
    monkeypatch.setattr(sys, "argv", argv)
    main.main()


def test_copy_subset_key_absent_from_config_is_allowed(tmp_path, monkeypatch):
    config_path = write_copy_dataset(tmp_path)
    inp = tmp_path / "input" / "{subject}" / "{frame}.png"
    out = tmp_path / "output" / "{subject}" / "{frame}.png"
    run_main(monkeypatch, "copy", inp, out, config_path, "subject=alice")
    assert (tmp_path / "output" / "alice" / "left.png").exists()


def test_copy_subset_key_absent_from_input_template_is_rejected(tmp_path, monkeypatch):
    config_path = write_copy_dataset(tmp_path)
    inp = tmp_path / "input" / "{subject}" / "{frame}.png"
    out = tmp_path / "output" / "{subject}" / "{frame}.png"
    with pytest.raises(ValueError, match="Keys available to subset on are"):
        run_main(monkeypatch, "copy", inp, out, config_path, "traj=0")


def test_pack_still_rejects_subset_key_absent_from_config(tmp_path, monkeypatch):
    config_path = write_copy_dataset(tmp_path)
    with pytest.raises(ValueError, match="traj"):
        run_main(
            monkeypatch,
            "pack",
            tmp_path / "input",
            tmp_path / "output",
            config_path,
            "traj=0",
        )
