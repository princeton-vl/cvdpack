import fnmatch
import itertools
import json
import logging
import re
import shutil
from glob import escape as escape_glob
from pathlib import Path
from urllib.parse import unquote

import huggingface_hub

from cvdpack import util

logger = logging.getLogger("cvdpack")


def parse_input(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"(?:https://huggingface\.co|hf:/)/datasets/([^/]+/[^/]+)"
        r"(?:/tree/([^/]+)(?:/(.*))?)?/?",
        value,
    )
    if match is None:
        if "://" in value or value.startswith("hf:"):
            raise ValueError(f"Unsupported --input URL: {value}")
        return None
    revision = unquote(match.group(2)) if match.group(2) else "main"
    subpath = unquote(match.group(3) or "")
    return match.group(1), revision, subpath


def _download_pattern(template: str, values: dict) -> str:
    def replace(match: re.Match) -> str:
        field = match.group(1)
        key = field.partition(":")[0]
        if key not in values:
            return "*"
        formatted = util.format_template(match.group(0), {key: values[key]})
        return escape_glob(str(formatted))

    return re.sub(r"\{([^}]+)\}", replace, template)


def _selected_templates(config: dict, subset: dict) -> dict[str, str]:
    return {
        gt_type: data_type["packed_path_template"]
        for gt_type, data_type in config["data_types"].items()
        if gt_type in util.as_list(subset.get("gt_type", gt_type))
    }


def _subset_combinations(gt_type: str, template: str, subset: dict) -> list[dict]:
    keys = [
        key
        for key in subset
        if key != "gt_type"
        and re.search(r"\{" + re.escape(key) + r"(?::[^}]*)?\}", template)
    ]
    values = itertools.product(*(util.as_list(subset[key]) for key in keys))
    return [{"gt_type": gt_type, **dict(zip(keys, combo))} for combo in values]


def download_patterns(config: dict, subset: dict, prefix: str) -> list[str]:
    allowed = {"gt_type"}
    for data_type in config["data_types"].values():
        allowed |= util.template_fields(data_type["packed_path_template"])
    util.validate_subset_keys(subset, allowed)

    patterns = set()
    for gt_type, template in _selected_templates(config, subset).items():
        for selected in _subset_combinations(gt_type, template, subset):
            patterns.add(escape_glob(prefix) + _download_pattern(template, selected))
    return sorted(patterns)


def select_config_file(repo_files: list[str], prefix: str) -> str:
    candidates = ["cvdpack.json"]
    if prefix:
        candidates.append(f"{prefix}cvdpack.json")
    for candidate in candidates:
        if candidate in repo_files:
            return candidate
    raise FileNotFoundError(
        f"Found no cvdpack.json at any of {candidates}, use --config to supply one"
    )


def _matches_pattern(file: str, pattern: str) -> bool:
    file_parts = file.split("/")
    pattern_parts = pattern.split("/")
    if len(file_parts) != len(pattern_parts):
        return False
    return all(
        fnmatch.fnmatchcase(file_part, pattern_part)
        for file_part, pattern_part in zip(file_parts, pattern_parts)
    )


def matching_files(
    repo_files: list[str], patterns: list[str], prefix: str = ""
) -> list[str]:
    return [
        file
        for file in repo_files
        if file.startswith(prefix)
        if any(_matches_pattern(file, pattern) for pattern in patterns)
    ]


# globs cannot express repeated-field equality; fill subset values first, they may contain "/"
def layout_regexes(config: dict, subset: dict, prefix: str) -> list[re.Pattern]:
    regexes = []
    for gt_type, template in _selected_templates(config, subset).items():
        for selected in _subset_combinations(gt_type, template, subset):
            filled = util.format_template(template, selected)
            regexes.append(util.template_to_regex(prefix + str(filled)))
    return regexes


def layout_files(files: list[str], regexes: list[re.Pattern]) -> list[str]:
    return [f for f in files if any(r.match(f) for r in regexes)]


# zero-byte placeholders let job discovery run before any data is downloaded
def stage_placeholders(dest: Path, files: list[str]) -> None:
    for file in files:
        path = dest / file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()


def fetch_job_input(remote: dict, input_path: Path) -> None:
    dest = Path(remote["dest"])
    file = input_path.relative_to(dest).as_posix()
    logger.info(f"Downloading {file} from {remote['repo_id']}")
    # the hub client may skip an existing destination, so drop the placeholder
    input_path.unlink(missing_ok=True)
    huggingface_hub.hf_hub_download(
        remote["repo_id"],
        file,
        repo_type="dataset",
        revision=remote["revision"],
        local_dir=dest,
    )


def retain_config(config_path: Path, data_folder: Path) -> Path:
    retained = data_folder / "cvdpack.json"
    if config_path.resolve() == retained.resolve():
        return config_path
    data_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, retained)
    logger.info(f"Retained config for {data_folder} as {retained}")
    return retained


def download_input(
    source: tuple[str, str, str],
    dest: Path,
    config_path: Path | None,
    subset: dict | None,
    expected_compatibility_version: int,
    staging: str = "upfront",
) -> tuple[Path, Path, dict | None]:
    repo_id, revision, subpath = source
    prefix = f"{subpath.strip('/')}/" if subpath else ""
    hub_args = dict(repo_type="dataset", revision=revision, local_dir=dest)
    repo_files = huggingface_hub.list_repo_files(
        repo_id, repo_type="dataset", revision=revision
    )

    # only a root repository config describes paths relative to the repository root
    template_prefix = prefix
    if config_path is None:
        config_file = select_config_file(repo_files, prefix)
        logger.info(f"Using config {config_file} from {repo_id}")
        config_path = Path(
            huggingface_hub.hf_hub_download(repo_id, config_file, **hub_args)
        )
        if config_file == "cvdpack.json":
            template_prefix = ""

    config = json.loads(config_path.read_text())
    config_compatibility = config.get("metadata", {}).get("compatibility_version")
    if (
        config_compatibility is not None
        and config_compatibility != expected_compatibility_version
    ):
        raise ValueError(
            f"Config {config_path} had compatibility version {config_compatibility}, "
            f"which differs from installed {expected_compatibility_version}"
        )
    patterns = download_patterns(config, subset or {}, template_prefix)
    selected = matching_files(repo_files, patterns, prefix)
    regexes = layout_regexes(config, subset or {}, template_prefix)
    selected = layout_files(selected, regexes)
    subpath_hint = (
        f". The url subpath {subpath!r} only narrows whole template components; "
        "select a field value containing '/' with --subset instead"
    )
    if len(selected) == 0:
        msg = f"No files in {repo_id} matched any of {patterns}, check --subset against the dataset layout"
        if prefix and not template_prefix:
            msg += subpath_hint
        raise ValueError(msg)

    remote = None
    if staging == "per_job":
        logger.info(f"Deferring download of {len(selected)} files to workers")
        stage_placeholders(dest, selected)
        remote = {"repo_id": repo_id, "revision": revision, "dest": str(dest)}
    else:
        logger.info(f"Downloading {len(selected)} files from {repo_id}")
        huggingface_hub.snapshot_download(
            repo_id,
            allow_patterns=[escape_glob(file) for file in selected],
            **hub_args,
        )
    data_root = dest / subpath if template_prefix else dest
    return data_root, retain_config(config_path, data_root), remote
