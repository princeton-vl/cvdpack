import pytest

from cvdpack import util

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
    util.validate_subset_keys({"scene": "a", "cam": "left"}, CONFIG)
    util.validate_subset_keys({"gt_type": "rgb"}, CONFIG)


def test_validate_subset_keys_accepts_key_missing_from_one_template():
    util.validate_subset_keys({"gt_type": "metadata", "cam": "left"}, CONFIG)


def test_validate_subset_keys_rejects_unknown_key():
    with pytest.raises(ValueError, match="traj"):
        util.validate_subset_keys({"scene": "a", "traj": "0"}, CONFIG)


def test_validate_subset_keys_error_lists_available_keys():
    with pytest.raises(ValueError, match="scene"):
        util.validate_subset_keys({"traj": "0"}, CONFIG)


def test_validate_subset_keys_noop_without_config_or_subset():
    util.validate_subset_keys({"traj": "0"}, None)
    util.validate_subset_keys(None, CONFIG)
    util.validate_subset_keys({}, CONFIG)
