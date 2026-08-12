import json
import sys
from pathlib import Path

from cvdpack.util import match_template_paths


def main() -> None:
    config_path, src, unpacked = sys.argv[1:4]
    config = json.loads(Path(config_path).read_text())

    for gt_type, spec in config["data_types"].items():
        template = spec.get("original_path_template")
        if template is None:
            continue
        template = template.replace("{gt_type}", gt_type)
        n_src = sum(1 for _ in match_template_paths(Path(src) / template))
        n_out = sum(1 for _ in match_template_paths(Path(unpacked) / template))
        print(f"{gt_type}\t{template}\t{n_src}\t{n_out}")


if __name__ == "__main__":
    main()
