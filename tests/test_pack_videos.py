import logging
import tarfile
from pathlib import Path

import numpy as np
import pytest

import cvdpack.util as util
from cvdpack.pack_timeseries import (
    pack_tarball,
    pack_video,
    unpack_tarball,
    unpack_video,
)
from cvdpack.util import load_any_image, save_any_image

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3])
def test_video_pack_unpack_roundtrip(tmp_path, dtype, channels):
    np.random.seed(42)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    unpacked_dir = tmp_path / "unpacked"
    unpacked_dir.mkdir()

    n_frames = 5
    max_val = np.iinfo(dtype).max
    shape = (40, 32) if channels == 1 else (40, 40, channels)
    original_frames = []
    for i in range(n_frames):
        data = np.random.randint(0, max_val + 1, shape, dtype=dtype)
        original_frames.append(data.copy())
        frame_path = frames_dir / f"frame_{i:04d}.png"
        save_any_image(data, frame_path)

    video_path = tmp_path / "test_video.mkv"
    pack_video(
        input_frames_path=frames_dir / "frame_{frame:04d}.png",
        output_video_path=video_path,
        tmp_folder=tmp_path / "tmp",
        loglevel=logging.DEBUG,
    )
    assert video_path.exists(), f"Video file {video_path} was not created"

    out_template = unpacked_dir / "frame_{frame:04d}.png"
    unpack_video(
        input_video_path=video_path,
        output_frames_path_template=out_template,
        tmp_folder=tmp_path / "tmp",
        loglevel=logging.DEBUG,
    )

    out_paths = list(util.match_template_paths(out_template))
    if len(out_paths) != len(original_frames):
        raise ValueError(
            f"Expected {len(original_frames)=} but got {len(out_paths)=} "
            f"got {out_paths=}"
        )

    # Verify equality
    for i in range(len(original_frames)):
        original = original_frames[i]
        unpacked_info, unpacked_path = out_paths[i]
        unpacked = load_any_image(unpacked_path)
        np.testing.assert_array_equal(unpacked, original)


def test_tarball_pack_unpack_roundtrip(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    expected: dict[str, bytes] = {}
    for i in range(3):
        name = f"frame_{i:04d}.txt"
        data = f"frame {i}\n".encode()
        (frames / name).write_bytes(data)
        expected[name] = data

    archive = tmp_path / "frames.tar.gz"
    pack_tarball(frames / "frame_{frame:04d}.txt", archive)

    unpacked = tmp_path / "unpacked"
    unpack_tarball(archive, unpacked / "frame_{frame:04d}.txt")

    actual = {path.name: path.read_bytes() for path in unpacked.iterdir()}
    assert actual == expected


def test_tarball_unpack_flattens_a_folder_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "Image.tar.gz"
    expected = {f"Image_{i:04d}.png": f"frame {i}\n".encode() for i in range(3)}

    # Upstream releases (e.g. InFlux-Synth) store members under a folder prefix
    staging = tmp_path / "staging" / "Image"
    staging.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in expected.items():
            (staging / name).write_bytes(data)
            tar.add(staging / name, arcname=f"Image/{name}")

    unpacked = tmp_path / "unpacked" / "Image"
    unpack_tarball(archive, unpacked / "Image_{frame:04d}.png")

    assert {p.name: p.read_bytes() for p in unpacked.rglob("*") if p.is_file()} == (
        expected
    )
    assert not (unpacked / "Image").exists()


def test_tarball_unpack_refuses_to_drop_colliding_frames(tmp_path: Path) -> None:
    archive = tmp_path / "Image.tar.gz"
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"x")

    # the pre-fix pack_tarball named every member after the tarball itself
    with tarfile.open(archive, "w:gz") as tar:
        for _ in range(3):
            tar.add(frame, arcname="Image.tar.gz")

    with pytest.raises(ValueError, match="collapse to"):
        unpack_tarball(archive, tmp_path / "unpacked" / "Image_{frame:04d}.png")


def test_tarball_unpack_ignores_traversal_in_member_names(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"pwned\n")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../../escaped.txt")

    unpacked = tmp_path / "unpacked" / "Image"
    unpack_tarball(archive, unpacked / "Image_{frame:04d}.png")

    assert (unpacked / "escaped.txt").read_bytes() == b"pwned\n"
    assert not (tmp_path / "escaped.txt").exists()
