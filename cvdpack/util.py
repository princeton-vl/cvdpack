import re
from pathlib import Path
from string import Formatter
from typing import Generator

import cv2
import numpy as np
import logging

logger = logging.getLogger("cvdpack")

ENVIRON_KEYS = {
    "ffmpeg": "CVDPACK_FFMPEG",
    "array_max": "CVDPACK_SLURM_ARRAY_MAX",
    "ffv1_args": "CVDPACK_FFV1_ARGS",
    "libx265_args": "CVDPACK_LIBX265_ARGS",
    "allow_lossy_rgb_encode": "CVDPACK_MINOR_VIDEO_ERROR",
}


def load_any_image(path: Path, allow_pickle: bool = False):
    match path.suffix:
        case ".png" | ".jpg" | ".jpeg":
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        case ".npy":
            return np.load(path, allow_pickle=allow_pickle)
        case ".npz":
            return dict(np.load(path, allow_pickle=allow_pickle))
        case ".exr":
            return cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        case _:
            raise ValueError(f"Unhandled {path.suffix=} for {path=}")


def save_any_image(
    img: np.ndarray,
    path: Path,
):
    logger.debug(f"Saving {img.shape=} {img.dtype=} to {path=}")

    match path.suffix, img.dtype:
        case ((".png" | ".jpg" | ".jpeg"), np.uint8 | np.uint16):
            if img.ndim == 3 and img.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"Unhandled {img.shape=} for {path=}, expected no channels (WxH) or 3 channels (WxHx3) or 4 channels (WxHx4)"
                    "These correspond to grayscale, RGB or RGBA images. But no format exists for 2channel or 5+channel"
                )
            cv2.imwrite(str(path), img)
        case ".npy", _:
            np.save(path, img)
        case _:
            raise ValueError(f"Unhandled {path.suffix=} {img.dtype=}")

    assert path.exists(), f"Failed to save {path=}"


def template_to_regex(template: Path, allow_any: list[str] | None = None):
    fmt = Formatter()

    found_keys = set()
    parts = []
    for lit, field, conv, _ in fmt.parse(template):
        if "*" in lit:
            lit_parts = lit.split("*")
            for i, part in enumerate(lit_parts):
                if i > 0:
                    parts.append(r"[^/\\]*")
                parts.append(re.escape(part))
        else:
            parts.append(re.escape(lit))

        if not field:
            continue

        if isinstance(conv, str) and conv.endswith("d") and field not in allow_any:
            restrictor = r"\d+"
        else:
            restrictor = r"[^/\\]+"

        if field in found_keys:
            part = rf"{restrictor}"
        else:
            part = rf"(?P<{field}>{restrictor})"

        parts.append(part)
        found_keys.add(field)

    regex = "^" + "".join(parts) + "$"
    try:
        return re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid regex: {regex=}, {e=}") from e


def match_template_paths(
    template: Path,
    match_video_folder: bool = False,
    allow_any: list[str] | None = None,
) -> Generator[tuple[dict, Path], None, None]:
    if allow_any is None:
        allow_any = []

    first_curlypart = next((i for i, p in enumerate(template.parts) if "{" in p), None)
    if first_curlypart is None:
        if template.exists():
            yield ({}, template)
            return
        raise ValueError(f"{template=} has no {{}}. Nothing to match?")
    child_template = "/".join(template.parts[first_curlypart:])
    search_folder = Path(*template.parts[:first_curlypart])

    if not search_folder.exists():
        raise ValueError(
            f"{child_template=} has base {search_folder=} which does not exist"
        )

    regex = template_to_regex(child_template, allow_any=allow_any)

    def match_to_dict(m: re.Match):
        return {k: int(v) if v.isdigit() else v for k, v in m.groupdict().items()}

    glob_pattern = re.sub(r"\{[^}]*\}", "*", child_template)

    logger.debug(
        f"Searching {search_folder=} using {glob_pattern=} created from {child_template=}"
    )
    files = sorted(list(search_folder.rglob(glob_pattern)))
    logger.debug(
        f"{search_folder=} had {len(files)} files matching {glob_pattern=}, testing against {regex=}"
    )
    for p in files:
        teststr = str(p.relative_to(search_folder))
        m = regex.match(teststr)
        if m:
            yield match_to_dict(m), p


def format_template(
    template: Path,
    vals: dict,
    allow_missing: list[str] | None = None,
    return_matched: bool = False,
) -> Path | tuple[Path, dict]:
    """
    Args:
        template: Path or str, must contain {field} or {field:...d} style template strings
        vals: dict, keys must match the {field} strings
        allow_missing: list[str] | None - if provided, keys in the template but not in this list will raise an error
        return_matched: bool - if True, return the matched keys
    """

    matched = set()

    def replace_func(match):
        full_spec = match.group(1)
        key = full_spec.split(":")[0]
        if key in vals:
            try:
                res = ("{" + full_spec + "}").format(**{key: vals[key]})
                matched.add(key)
                return res
            except ValueError as e:
                raise ValueError(
                    f"Invalid {full_spec=} for {key=} {vals[key]=} in {template=}, {e=}"
                ) from e
            except KeyError as e:
                raise ValueError(
                    f"Missing {key=} in {vals=} for {template=}, {allow_missing=}"
                ) from e
        elif allow_missing and key not in allow_missing:
            raise ValueError(
                f"Missing {key=} in {vals=} for {template=}, {allow_missing=}"
            )

        return match.group(0)

    res = re.sub(r"\{([^}]+)\}", replace_func, str(template))

    if isinstance(template, Path):
        res = Path(res)

    # logger.debug(f"{format_template.__name__} {template=} -> {res=}, {matched=}")

    if return_matched:
        return res, matched
    return res


def parse_dictlist_strings(argstrings: list[str] | None):
    if argstrings is None:
        return None

    args = {}
    for arg in argstrings:
        parts = arg.split("=")
        if len(parts) != 2:
            raise ValueError(f"Invalid {arg=}, had {len(parts)=}")
        k, v = parts
        if "," in v:
            v = list(v.split(","))
        args[k] = v

    logger.debug(f"{parse_dictlist_strings.__name__} mapped {argstrings=} -> {args=}")
    return args


def included_in_filter(
    file_keys: dict,
    filter_vals: dict | None,
    allow_extra: set[str] | None = None,
) -> bool:
    if filter_vals is None:
        return False

    first_keys = set(file_keys.keys())
    extra = set(filter_vals.keys()) - first_keys
    if allow_extra is not None:
        extra -= allow_extra
    if extra:
        raise ValueError(
            f"{filter_vals=} had keys {extra} which are not present in the input file template. "
            f"Keys available to filter on are {first_keys}"
        )

    res = all(
        (
            k not in file_keys
            or file_keys[k] == v
            or (isinstance(v, (list, set)) and file_keys[k] in v)
        )
        for k, v in filter_vals.items()
    )
    return res
