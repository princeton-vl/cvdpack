from enum import Enum
import numpy as np
import logging
from typing import Literal, Any
from pathlib import Path

from .util import match_template_paths, format_template, load_any_image, save_any_image

logger = logging.getLogger("cvdpack")

DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


class PackMethod(Enum):
    LINEAR = "linear"
    INV = "inv"
    F32_AS_2INT16 = "f32_as_2int16"
    F16_AS_INT16 = "f16_as_int16"
    CHECKBOUNDS = "checkbounds"

    @classmethod
    def from_str(cls, s: str):
        return cls(s.lower())


def _mask_to_nan_or_error(
    img: np.ndarray,
    mask: np.ndarray,
    oob_method: Literal["nan", "nan_warn", "error"],
    msg: str,
):
    match oob_method:
        case "nan":
            img[mask] = np.nan
        case "nan_warn":
            img[mask] = np.nan
            logger.warning(msg + ", will be interpreted as nan")
        case "error":
            raise ValueError(msg)


def _oob_to_nan_or_error(
    img: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    oob_method: Literal["nan", "nan_warn", "error"],
):
    oob_mask = np.logical_or(img < min_orig_val, img > max_orig_val)
    if not oob_mask.any():
        return img

    oob_pct = 100 * oob_mask.astype(np.float32).mean()
    msg = (
        f"Found {img.min()=:.2f}, {img.max()=:.2f} "
        f"which exceeds quantize range [{min_orig_val:.2f}, {max_orig_val:.2f}]. {oob_pct:.2f}% were out of bounds."
    )

    img = img.copy()
    _mask_to_nan_or_error(img, oob_mask, oob_method, msg)
    return img


def _pack_to_int_with_nan_to_imax(
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


def _unpack_from_int_with_imax_to_nan(
    img_quant: np.ndarray,
    from_dtype: np.dtype,
):
    assert np.issubdtype(from_dtype, np.integer)
    imax = np.iinfo(from_dtype).max
    img_norm = img_quant.astype(np.float64) / (imax - 1)
    img_norm[img_quant == imax] = np.nan
    return img_norm


class Packer:
    def pack(self, img: np.ndarray):
        raise NotImplementedError("Subclass must implement pack")

    def unpack(self, img_packed: np.ndarray):
        raise NotImplementedError("Subclass must implement unpack")


class LinearQuantizeIntPacker(Packer):
    """
    We use uint8/uint16 since pngs/mkvs support them
    """

    def __init__(
        self,
        min_orig_val: float,
        max_orig_val: float,
        from_dtype: np.dtype,
        to_dtype: np.dtype,
        out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
    ):
        self.min_orig_val = min_orig_val
        self.max_orig_val = max_orig_val
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        self.out_of_bounds_method = out_of_bounds_method

        assert to_dtype in [np.uint8, np.uint16], f"{to_dtype=}"

    def pack(self, img: np.ndarray):
        img = _oob_to_nan_or_error(
            img,
            self.min_orig_val,
            self.max_orig_val,
            self.out_of_bounds_method,
        )
        img_norm = (img - self.min_orig_val) / (self.max_orig_val - self.min_orig_val)
        return _pack_to_int_with_nan_to_imax(img_norm, self.to_dtype)

    def unpack(self, img_packed: np.ndarray):
        img_norm = _unpack_from_int_with_imax_to_nan(img_packed, self.to_dtype)
        img_orig = (
            img_norm * (self.max_orig_val - self.min_orig_val) + self.min_orig_val
        )
        return img_orig.astype(self.from_dtype)


class InvQuantizeInt16Packer(Packer):
    """
    Convert to uint8/uint16 but with nonlinear 1/x mapping, i.e. greater precision for small values

    TODO: we may be able to generalize this for more mapping functions e.g. syssqrt / log?
        but we would need to be aware/handle how they flip the min/max or crash <0 etc
    """

    def __init__(
        self,
        min_orig_val: float,
        max_orig_val: float,
        from_dtype: np.dtype,
        to_dtype: np.dtype,
        out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
    ):
        assert max_orig_val > 0, max_orig_val
        assert min_orig_val > 0, min_orig_val

        self.linear_packer = LinearQuantizeIntPacker(
            min_orig_val=1
            / max_orig_val,  # swapped min and max because of 1/x mapping!
            max_orig_val=1 / min_orig_val,
            from_dtype=from_dtype,
            to_dtype=to_dtype,
            out_of_bounds_method=out_of_bounds_method,
        )

    def pack(self, img: np.ndarray):
        return self.linear_packer.pack(1 / img)

    def unpack(self, img_quant: np.ndarray):
        return 1 / self.linear_packer.unpack(img_quant)


class F32As2Int16ReinterpretPacker(Packer):
    """
    Reinterprets one f32 channel into 2 uint16 image channels.

    Unfortunately the f32 exponent / mantissa do not neatly align with 16bits.

    Currently we just ignore and hope the resulting images compress ok.

    TODO: we could split into 3 uint16 channels with better alignment? does this matter?
    """

    def __init__(self, scalar: float):
        self.scalar = scalar

    def pack(self, img: np.ndarray):
        if img.ndim == 2:
            img = img[..., np.newaxis]
        if img.dtype != np.float32:
            raise ValueError(
                f"Expected float32 for {PackMethod.F32_AS_2INT16=}, got {img.dtype=}, {img.shape=}"
            )
        if img.shape[2] > 2:
            raise ValueError(
                f"Expected 1 or 2 channels for {PackMethod.F32_AS_2INT16=}, got {img.shape=} "
                "This is beacuse 2xf32 will become 4xuint16, any more than this cannot fit into RGBA uint16"
            )

        # use view to reinterpret float bytes as uint16.
        # this adds a 2 channel, or multiplies the last channel by 2, which is what we want anyway
        assert img.dtype == np.float32, img.dtype
        img_quant = img.view(np.uint16)

        return img_quant

    def unpack(self, img_packed: np.ndarray):
        if (
            img_packed.shape[2] == 3
        ):  # png saving will add an extra channel to fake that it is an RGB
            img_packed = img_packed[..., :2]
        assert img_packed.dtype == np.uint16
        img = img_packed.view(dtype=np.float32)
        return img


class F16ToInt16ReinterpretPacker(Packer):
    """
    Casts every f32 as f16, then reinterprets to uint16.

    Since we are losing precision, it may be useful to multiply by a scalar then undo it later, so we can have
    more precision but over a smaller dynamic range. If this overflows, we will nan the value and warn/error.

    We would prefer to just reinterpret as uint32, but to my knowledge there are no 32bit-channel png/mkv formats (2025-07-20)
    """

    def __init__(
        self,
        scalar: float,
        out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "error",
    ):
        self.scalar = scalar
        self.out_of_bounds_method = out_of_bounds_method

    def pack(self, img: np.ndarray):
        mult = (img * self.scalar).astype(np.float16)

        # check for overflow - new inf values which were not present before multiply
        newinf = np.isinf(mult) * np.isfinite(img)
        if newinf.any():
            msg = (
                f"{self.__class__.__name__} had {newinf.astype(np.float32).mean()=}% "
                f"this likely means that {self.scalar=} * {np.abs(img).max()=} is too large and caused overflow for float16"
            )
            _mask_to_nan_or_error(mult, newinf, self.out_of_bounds_method, msg)

        return mult.view(dtype=np.uint16)  # reinterpret cast

    def unpack(self, img_packed: np.ndarray):
        if img_packed.ndim == 2:
            img_packed = img_packed[..., np.newaxis]
        assert img_packed.shape[2] <= 3
        assert img_packed.dtype == np.uint16
        img = img_packed.view(dtype=np.float16)  # reinterpret cast
        return img


class CheckBoundsPacker(Packer):
    def __init__(
        self,
        min_orig_val: float,
        max_orig_val: float,
        to_dtype: np.dtype,
        from_dtype: np.dtype,
    ):
        self.min_orig_val = min_orig_val
        self.max_orig_val = max_orig_val
        self.to_dtype = to_dtype
        self.from_dtype = from_dtype

    def pack(self, img: np.ndarray):
        assert img.dtype == self.from_dtype, f"{img.dtype=}, {self.from_dtype=}"
        img = _oob_to_nan_or_error(
            img,
            self.min_orig_val,
            self.max_orig_val,
            "error",
        )
        img_quant = img.astype(self.to_dtype)
        return img_quant

    def unpack(self, img_packed: np.ndarray):
        assert img_packed.dtype == self.to_dtype, (
            f"{img_packed.dtype=}, {self.to_dtype=}"
        )
        return img_packed.astype(self.from_dtype)


def get_channel_packer(packing_config: dict[str, Any]) -> Packer:
    min_orig_val = packing_config.get("min_orig_val", None)
    max_orig_val = packing_config.get("max_orig_val", None)
    to_dtype = (
        DTYPE_MAP[packing_config["to_dtype"]] if "to_dtype" in packing_config else None
    )
    from_dtype = (
        DTYPE_MAP[packing_config["from_dtype"]]
        if "from_dtype" in packing_config
        else None
    )
    out_of_bounds_method = packing_config.get("out_of_bounds_method")

    match PackMethod.from_str(packing_config["method"]):
        case PackMethod.LINEAR:
            return LinearQuantizeIntPacker(
                min_orig_val=min_orig_val,
                max_orig_val=max_orig_val,
                from_dtype=from_dtype,
                to_dtype=to_dtype,
                out_of_bounds_method=out_of_bounds_method,
            )
        case PackMethod.INV:
            return InvQuantizeInt16Packer(
                min_orig_val=min_orig_val,
                max_orig_val=max_orig_val,
                from_dtype=from_dtype,
                to_dtype=to_dtype,
                out_of_bounds_method=out_of_bounds_method,
            )
        case PackMethod.F32_AS_2INT16:
            return F32As2Int16ReinterpretPacker(
                scalar=packing_config.get("scalar", 1.0),
            )
        case PackMethod.F16_AS_INT16:
            return F16ToInt16ReinterpretPacker(
                scalar=packing_config.get("scalar", 1.0),
                out_of_bounds_method=out_of_bounds_method,
            )
        case PackMethod.CHECKBOUNDS:
            return CheckBoundsPacker(
                min_orig_val=min_orig_val,
                max_orig_val=max_orig_val,
                to_dtype=to_dtype,
                from_dtype=from_dtype,
            )
        case _:
            raise ValueError(f"Invalid {packing_config['method']=}")


def get_all_channel_packers(
    channel_configs: dict[str, Any],
) -> dict[str, Packer | None]:
    return {
        channel_name: (
            get_channel_packer(conf)
            if (conf := channel_config.get("packing")) is not None
            else None
        )
        for channel_name, channel_config in channel_configs.items()
    }


def pack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    packer: Packer,
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
            img_quant = packer.pack(img)
        except Exception as e:
            raise ValueError(
                f"Error packing {frame_input_path=} to {output_path=}: {e}"
            ) from e

        if img_quant.ndim == 3 and img_quant.shape[2] == 2:
            # last dim 1 or 3 is fine, but 2 needs padding to 3 because 2-channel pngs are not a thing (?)
            img_quant = np.pad(img_quant, ((0, 0), (0, 0), (0, 1)), mode="constant")
            assert img_quant.shape[2] == 3, f"{img_quant.shape=}"

        save_any_image(img_quant, output_path)


def unpack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    packer: Packer,
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

        img_unquant = packer.unpack(img)

        match unpack_channels_last:
            case x if x is not None:
                assert img_unquant.ndim == 3, img_unquant.shape
                img_unquant = img_unquant[:, :, :x]
            case None if img_unquant.ndim == 3 and img_unquant.shape[2] == 1:
                img_unquant = img_unquant[:, :, 0]
            case None:
                pass
            case _:
                raise ValueError(f"{unpack_channels_last=} is not valid")

        if "frame" in frame_info:
            frame_info["framenext"] = frame_info["frame"] + 1
        output_img_path = format_template(
            output_path_template, frame_info, allow_missing=["framenext"]
        )
        logger.debug(
            f"Unquantizing {input_img_path=} to {output_img_path=}, {img_unquant.min()=}, {img_unquant.max()=}"
        )
        save_any_image(img_unquant, output_img_path)
