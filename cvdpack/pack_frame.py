from enum import Enum
import numpy as np
import logging
from typing import Literal
from pathlib import Path

from .util import match_template_paths, format_template, load_any_image, save_any_image

logger = logging.getLogger("cvdpack")


class PackMethod(Enum):
    LINEAR = "linear"
    INV = "inv"
    ONECHANNEL_F32_AS_2INT16 = "onechannel_f32_as_2int16"
    MULTICHANNEL_TO_F16_AS_INT16 = "multichannel_to_f16_as_int16"
    CHECKBOUNDS = "checkbounds"

    @classmethod
    def from_str(cls, s: str):
        return cls(s.lower())


def img_pack_to_orig(
    img_quant: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    to_dtype: np.dtype,
    pack_method: PackMethod,
    unpack_channels_last: int | None = None,
) -> np.ndarray:
    assert np.issubdtype(img_quant.dtype, np.unsignedinteger), f"{img_quant.dtype=}"
    imax = np.iinfo(img_quant.dtype).max
    quant_max = imax - 1  # exact maxint val is used for nan

    match pack_method:
        case PackMethod.CHECKBOUNDS:
            img = img_quant.astype(to_dtype)
        case PackMethod.LINEAR:
            img_norm = img_quant.astype(np.float64) / quant_max
            img_orig = img_norm * (max_orig_val - min_orig_val) + min_orig_val
            img = img_orig.astype(to_dtype)
            img[img_quant == imax] = np.nan
        case PackMethod.INV:
            assert max_orig_val > 0, max_orig_val
            assert min_orig_val > 0, min_orig_val
            img_norm = img_quant.astype(np.float64) / quant_max
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_unmap = img_norm * (max_norm - min_norm) + min_norm
            img_orig = 1 / img_unmap
            img = img_orig.astype(to_dtype)
            img[img_quant == imax] = np.nan
        case PackMethod.ONECHANNEL_F32_AS_2INT16:
            # shape should be HxWx2 where 2 unpacks to first16bits, second16bits
            assert img_quant.shape[2] == 2, img_quant.shape
            assert img_quant.dtype == np.uint16
            img = img_quant.view(dtype=np.float32)
        case PackMethod.MULTICHANNEL_TO_F16_AS_INT16:
            if img_quant.ndim == 2:
                img_quant = img_quant[..., np.newaxis]
            assert img_quant.shape[2] <= 3
            assert img_quant.dtype == np.uint16
            img = img_quant.view(dtype=np.float16)  # reinterpret cast
        case _:
            raise ValueError(f"Invalid {pack_method=}")

    if unpack_channels_last is not None:
        # needed for cases like flow, which can be 2 channel, but will have been promoted to a 3 channel png/mkv
        assert img.ndim == 3, f"{img.ndim=}"
        assert img.shape[-1] >= unpack_channels_last, f"{img.shape=}"
        img = img[..., :unpack_channels_last]

    return img


def _oob_to_nan_or_error(
    img: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    oob_method: Literal["nan", "nan_warn", "error"],
    desc: str,
):
    oob_mask = np.logical_or(img < min_orig_val, img > max_orig_val)
    if not oob_mask.any():
        return

    oob_pct = 100 * oob_mask.astype(np.float32).mean()
    msg = (
        f"file {desc} had {img.min()=:.2f}, {img.max()=:.2f} "
        f"which exceeds quantize range [{min_orig_val:.2f}, {max_orig_val:.2f}]. {oob_pct:.2f}% were out of bounds."
    )

    match oob_method:
        case "nan":
            img[oob_mask] = np.nan
        case "nan_warn":
            img[oob_mask] = np.nan
            logger.warning(msg + ", will be interpreted as nan")
        case "error":
            raise ValueError(msg)


def _pack_nan_as_imax(
    img_norm: np.ndarray,
    to_dtype: np.dtype,
):
    assert np.issubdtype(to_dtype, np.integer)

    intmax = np.iinfo(to_dtype).max
    img_quant = np.zeros_like(img_norm, dtype=to_dtype)

    isnan = np.isnan(img_norm)
    img_quant[isnan] = intmax
    img_quant[~isnan] = (img_norm[~isnan] * (intmax - 1)).astype(to_dtype)
    return img_quant


def img_orig_to_pack(
    img: np.ndarray,
    to_dtype: np.dtype,
    pack_method: PackMethod,
    min_orig_val: float,
    max_orig_val: float,
    out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
    desc: str = "",
):
    """
    Args:
        desc: added to log warnings / errors to make them more descriptive, e.g. list the path
    """

    from_dtype = img.dtype

    assert np.issubdtype(to_dtype, np.integer)
    assert not np.issubdtype(to_dtype, np.signedinteger), (
        f"Cannot quantize to signed integer: {to_dtype=}"
    )

    match pack_method:
        case PackMethod.CHECKBOUNDS:
            assert np.issubdtype(from_dtype, np.integer)
            img_quant = img.astype(to_dtype)
        case PackMethod.LINEAR:
            img_norm = (img - min_orig_val) / (max_orig_val - min_orig_val)
            _oob_to_nan_or_error(
                img, min_orig_val, max_orig_val, out_of_bounds_method, desc
            )
            img_quant = _pack_nan_as_imax(img_norm, to_dtype)
        case PackMethod.INV:
            _oob_to_nan_or_error(
                img, min_orig_val, max_orig_val, out_of_bounds_method, desc
            )
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_norm = (1 / img - min_norm) / (max_norm - min_norm)
            img_quant = _pack_nan_as_imax(img_norm, to_dtype)
        case PackMethod.ONECHANNEL_F32_AS_2INT16:
            if img.ndim == 2:
                img = img[..., np.newaxis]
            W, H, D = img.shape
            if img.dtype != np.float32 or D != 1:
                raise ValueError(
                    f"Expected float32 1 channel for {PackMethod.ONECHANNEL_F32_AS_2INT16=}, got {img.dtype=}, {img.shape=}"
                )
            # view to reinterpret float bytes as uint16.
            # this puts an extra channel of dim 2 at the end, which is what we want.
            img_quant = img.view(np.float16).astype(np.uint16)  #
        case PackMethod.MULTICHANNEL_TO_F16_AS_INT16:
            assert img.dtype == np.float32, img.dtype
            img_quant = img.astype(np.float16).view(dtype=np.uint16)  # reinterpret cast
        case _:
            raise ValueError(f"Invalid {pack_method=}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Packing {img.shape=}, {img.dtype=}, "
            f"{img.min()=:.2f}, {img.max()=:.2f}, {img_quant.min()=:.2f}, {img_quant.max()=:.2f}"
        )

    return img_quant

def pack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: np.dtype,
    pack_method: PackMethod,
    min_orig_val: float,
    max_orig_val: float,
    out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
):
    logger.debug(
        f"{pack_frameset.__name__} {input_path_template=} to {output_path_template=}"
    )

    output_path_template.parent.mkdir(parents=True, exist_ok=True)

    all_files = list(match_template_paths(input_path_template))
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        img = load_any_image(frame_input_path)
        try:
            img_quant = img_orig_to_pack(
                img,
                to_dtype,
                pack_method=pack_method,
                min_orig_val=min_orig_val,
                max_orig_val=max_orig_val,
                out_of_bounds_method=out_of_bounds_method,
                desc=str(frame_input_path),
            )
        except Exception as e:
            raise ValueError(
                f"Error packing {frame_input_path=} to {output_path=}: {e}"
            ) from e

        if img.ndim == 3 and img.shape[2] == 2:
            # last dim 1 or 3 is fine, but 2 needs padding to 3 because 2-channel pngs are not a thing (?)
            img_quant = np.pad(img_quant, ((0, 0), (0, 0), (0, 1)), mode="constant")
            assert img_quant.shape[2] == 3, f"{img_quant.shape=}"

        save_any_image(img_quant, output_path)


def unpack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: np.dtype,
    pack_method: PackMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
):
    assert "{" in input_path_template.name, (
        f"Input path must contain a template: {input_path_template=}"
    )
    assert "{" in output_path_template.name, (
        f"Output path must contain a template: {output_path_template=}"
    )
    output_path_template.parent.mkdir(parents=True, exist_ok=True)
    all_files = match_template_paths(input_path_template)

    for frame_info, input_img_path in all_files:
        assert input_img_path.suffix == ".png"
        img = load_any_image(input_img_path)

        assert np.issubdtype(img.dtype, np.integer), (
            f"{input_img_path=} had {img.dtype=}"
        )

        if unpack_channels_last is not None:
            assert img.ndim == 3, img.shape
            img = img[:, :, :unpack_channels_last]

        img_unquant = img_pack_to_orig(
            img,
            min_orig_val,
            max_orig_val,
            to_dtype=to_dtype,
            pack_method=pack_method,
            unpack_channels_last=unpack_channels_last,
        )

        if "frame" in frame_info:
            frame_info["framenext"] = frame_info["frame"] + 1
        output_img_path = format_template(
            output_path_template, frame_info, allow_missing=["framenext"]
        )
        logger.debug(
            f"Unquantizing {input_img_path=} to {output_img_path=}, {img_unquant.min()=}, {img_unquant.max()=}"
        )
        save_any_image(img_unquant, output_img_path)