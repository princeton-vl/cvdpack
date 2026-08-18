from pathlib import Path

import pytest

from cvdpack.main import decide_dataset_job_templates

IN = Path("/in")
OUT = Path("/out")

VIDEO_PNG = {
    "original_path_template": "{scene}/frames/rgb/rgb_{frame:04d}.png",
    "packed_path_template": "{scene}/rgb.mkv",
}
VIDEO_NPY = {
    "original_path_template": "{scene}/depth/depth_{frame:04d}.npy",
    "packed_path_template": "{scene}/depth.mkv",
}
TXT_NPY = {
    "original_path_template": "{scene}/poses.txt",
    "packed_path_template": "{scene}/poses.npy",
}
NPZ_TAR = {
    "original_path_template": "{scene}/camview_{frame:04d}.npz",
    "packed_path_template": "{scene}/camview.tar.gz",
}


def _decide(conf: dict, steps: list[str] | None, mode: str) -> tuple[str, str]:
    inp, out = decide_dataset_job_templates(IN, OUT, conf, steps, mode)
    return str(inp), str(out)


def test_no_steps_uses_the_config_templates_both_ways() -> None:
    orig = "/in/{scene}/frames/rgb/rgb_{frame:04d}.png"
    packed = "/out/{scene}/rgb.mkv"
    assert _decide(VIDEO_PNG, None, "pack") == (orig, packed)

    orig_out = "/out/{scene}/frames/rgb/rgb_{frame:04d}.png"
    assert _decide(VIDEO_PNG, None, "unpack") == ("/in/{scene}/rgb.mkv", orig_out)


def test_pack_quantize_only_stops_at_pngs() -> None:
    inp, out = _decide(VIDEO_PNG, ["quantize"], "pack")
    assert inp == "/in/{scene}/frames/rgb/rgb_{frame:04d}.png"
    assert out == "/out/{scene}/frames/rgb/rgb_{frame:04d}.png"


def test_pack_video_only_starts_from_quantized_pngs() -> None:
    inp, out = _decide(VIDEO_PNG, ["pack_video"], "pack")
    assert inp == "/in/{scene}/frames/rgb/rgb_{frame:04d}.png"
    assert out == "/out/{scene}/rgb.mkv"


def test_unpack_unquantize_only_starts_from_pngs() -> None:
    inp, out = _decide(VIDEO_PNG, ["unquantize"], "unpack")
    assert inp == "/in/{scene}/frames/rgb/rgb_{frame:04d}.png"
    assert out == "/out/{scene}/frames/rgb/rgb_{frame:04d}.png"


def test_unpack_video_only_leaves_pngs_when_original_is_npy() -> None:
    inp, out = _decide(VIDEO_NPY, ["unpack_video"], "unpack")
    assert inp == "/in/{scene}/depth.mkv"
    assert out == "/out/{scene}/depth/depth_{frame:04d}.png"


def test_unpack_video_only_is_a_noop_when_original_is_already_png() -> None:
    inp, out = _decide(VIDEO_PNG, ["unpack_video"], "unpack")
    assert inp == "/in/{scene}/rgb.mkv"
    assert out == "/out/{scene}/frames/rgb/rgb_{frame:04d}.png"


def test_pack_video_from_txt_skips_the_txt_to_npy_conversion() -> None:
    inp, out = _decide(TXT_NPY, ["pack_video"], "pack")
    assert inp == "/in/{scene}/poses.npy"
    assert out == "/out/{scene}/poses.npy"


def test_unpack_video_to_txt_skips_the_npy_to_txt_conversion() -> None:
    inp, out = _decide(TXT_NPY, ["unpack_video"], "unpack")
    assert inp == "/in/{scene}/poses.npy"
    assert out == "/out/{scene}/poses.npy"


def test_unrelated_steps_leave_the_templates_alone() -> None:
    assert _decide(TXT_NPY, ["quantize"], "pack") == (
        "/in/{scene}/poses.txt",
        "/out/{scene}/poses.npy",
    )


def test_spelled_out_full_pack_equals_the_default() -> None:
    spelled = _decide(VIDEO_NPY, ["quantize", "pack_video"], "pack")
    assert spelled == _decide(VIDEO_NPY, None, "pack")


def test_spelled_out_full_unpack_equals_the_default() -> None:
    spelled = _decide(VIDEO_NPY, ["unpack_video", "unquantize"], "unpack")
    assert spelled == _decide(VIDEO_NPY, None, "unpack")


def test_single_stage_types_run_fully_for_any_steps() -> None:
    assert _decide(NPZ_TAR, ["quantize"], "pack") == _decide(NPZ_TAR, None, "pack")
    unpack = _decide(NPZ_TAR, ["unpack_video"], "unpack")
    assert unpack == _decide(NPZ_TAR, None, "unpack")


@pytest.mark.parametrize(
    "conf, steps, mode",
    [
        (VIDEO_PNG, ["unpack_video"], "pack"),
        (VIDEO_PNG, ["quantize"], "unpack"),
        (TXT_NPY, ["unpack_video"], "pack"),
    ],
)
def test_steps_from_the_wrong_direction_are_rejected(
    conf: dict, steps: list[str], mode: str
) -> None:
    with pytest.raises(ValueError, match="Unhandled"):
        _decide(conf, steps, mode)


@pytest.mark.parametrize(
    "steps",
    [["quantize", "unpack_video"], ["quantize", "quantise"], ["frobnicate"]],
)
def test_unknown_steps_mixed_with_valid_ones_are_rejected(steps: list[str]) -> None:
    with pytest.raises(ValueError, match="Unhandled"):
        _decide(VIDEO_PNG, steps, "pack")


def test_steps_valid_for_the_mode_but_not_this_type_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unhandled"):
        _decide(VIDEO_PNG, ["pack"], "pack")


def test_steps_out_of_chain_order_are_rejected() -> None:
    with pytest.raises(ValueError, match="out of order"):
        _decide(VIDEO_PNG, ["pack_video", "quantize"], "pack")


def test_empty_steps_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty --steps"):
        _decide(VIDEO_PNG, [], "pack")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        _decide(VIDEO_PNG, None, "sideways")
