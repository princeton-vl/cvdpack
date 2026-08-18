import json
from pathlib import Path

import pytest

from cvdpack.main import pack_dataset, unpack_dataset

CONFIG = {
    "metadata": {},
    "data_types": {
        "Image": {
            "original_path_template": "{scene}/frames/Image/Image_{frame}.png",
            "packed_path_template": "{scene}/Image.mkv",
            "packer": "linear",
        }
    },
}


@pytest.mark.parametrize("action", ["pack", "unpack"])
def test_empty_input_errors_instead_of_writing_nothing(
    tmp_path: Path, action: str
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"

    if action == "unpack":
        (src / "cvdpack.json").write_text(json.dumps(CONFIG))

    process = pack_dataset if action == "pack" else unpack_dataset
    with pytest.raises(ValueError, match="No data_type template matched"):
        process(
            src,
            out,
            steps=None,
            config=CONFIG,
            parallel_mode="none",
            slurm_args=None,
            n_workers=1,
            tmp_folder=tmp_path / "tmp",
            subset=None,
            lazy=False,
            missing_gt="silent",
        )

    assert not out.exists()
