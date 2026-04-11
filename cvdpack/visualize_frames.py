# Copyright (c) 2025, Princeton University
# This code is licensed under the BSD-3-Clause license provided in the root directory of this project.

# Visualization functions adapted from infinigen_v2/exporters/visualize_gt.py (BSD-3-Clause)
# Flow colorization adapted from flow_vis.py (MIT License, Tom Runia 2018)

import colorsys
import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from imageio import imwrite
from matplotlib import pyplot as plt

from cvdpack import util

logger = logging.getLogger("cvdpack")

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# ---------------------------------------------------------------------------
# Flow colorization (MIT License, Tom Runia 2018)
# ---------------------------------------------------------------------------


def make_colorwheel() -> np.ndarray:
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col += YG
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col += CB
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv: np.ndarray, clip_flow: Optional[float] = None, convert_to_bgr: bool = False) -> np.ndarray:
    assert flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u, v = flow_uv[:, :, 0], flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_data(input_path: Path) -> np.ndarray:
    s = str(input_path)
    if s.endswith(".npy"):
        return np.load(s)
    elif s.endswith(".exr"):
        return cv2.imread(s, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    raise ValueError(f"Unsupported format: {input_path}")


def load_single_channel(input_path: Path) -> np.ndarray:
    s = str(input_path)
    if s.endswith(".npy"):
        return np.load(s)
    elif s.endswith(".exr"):
        try:
            import OpenEXR
        except ImportError:
            raise ImportError("OpenEXR is required to load .exr files")
        f = OpenEXR.InputFile(s)
        channel, channel_type = next(iter(f.header()["channels"].items()))
        data = np.frombuffer(f.channel(channel, channel_type.type), np.float32)
        dw = f.header()["dataWindow"]
        sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        return data.reshape(sz)
    raise ValueError(f"Unsupported format: {input_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mask_to_transparency_checkerboard(img: np.ndarray, mask: np.ndarray, checker_size: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    y_indices, x_indices = np.ogrid[:h, :w]
    checkerboard = ((y_indices // checker_size) + (x_indices // checker_size)) % 2
    checker_color = np.where(checkerboard[..., None], 0.8, 0.6)
    if img.dtype == np.uint8:
        checker_color = (checker_color * 255).astype(np.uint8)
    img = img.copy()
    img = np.where(mask[..., None], checker_color, img)
    return img


def _imwrite(path: Path, img: np.ndarray) -> None:
    imwrite(path, img)


# ---------------------------------------------------------------------------
# Per-type visualizers
# ---------------------------------------------------------------------------


def visualize_flow(
    input_path: Path,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    flow = load_data(input_path)
    flow_uv = flow[..., :2]
    if vmin is not None and vmax is not None:
        flow_uv = (flow_uv - vmin) / (vmax - vmin)
    flow_color = flow_to_color(flow_uv, convert_to_bgr=False)
    flow_color = mask_to_transparency_checkerboard(flow_color, np.isnan(flow_uv).any(axis=2))
    _imwrite(output_path, flow_color)


def visualize_normals(input_path: Path, output_path: Path) -> None:
    normals = load_data(input_path)[..., [2, 0, 1]] * np.array([-1.0, 1.0, 1.0])
    norm = np.linalg.norm(normals, axis=2)
    color = np.round((normals + 1) * (255 / 2)).astype(np.uint8)
    color = mask_to_transparency_checkerboard(color, norm < 1e-4)
    _imwrite(output_path, color)


def visualize_depth(
    input_path: Path,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scale_vmin: float = 1.0,
) -> None:
    depth = load_single_channel(input_path)
    depth = 1 / depth
    depth_notnan = depth.copy()
    depth_notnan[np.isnan(depth_notnan)] = 0
    if vmin is None:
        vmin = depth.min() * scale_vmin
    if vmax is None:
        vmax = depth.max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth_colored = cmap(norm(depth_notnan))
    depth_colored = mask_to_transparency_checkerboard(depth_colored, np.isnan(depth_notnan))
    _imwrite(output_path, np.ascontiguousarray(depth_colored[..., :3] * 255, dtype=np.uint8))


def visualize_seg_mask(input_path: Path, output_path: Path, color_seed: int = 0) -> None:
    mask = load_single_channel(input_path).astype(np.int64)
    H, W = mask.shape
    data = mask.reshape((H * W, -1))
    uniq, indices = np.unique(data, return_inverse=True, axis=0)
    random_states = [np.random.RandomState(e[:2].astype(np.uint32) + color_seed) for e in uniq]
    unique_colors = (
        np.asarray([colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1) for s in random_states]) * 255
    ).astype(np.uint8)
    _imwrite(output_path, unique_colors[indices].reshape((H, W, 3)))


def visualize_uniq_inst(input_path: Path, output_path: Path, color_seed: int = 0) -> None:
    uniq_inst = load_data(input_path).view(np.int32)
    H, W = uniq_inst.shape[:2]
    data = uniq_inst.reshape((H * W, -1))
    uniq, indices = np.unique(data, return_inverse=True, axis=0)
    random_states = [np.random.RandomState(e[:2].astype(np.uint32) + color_seed) for e in uniq]
    unique_colors = (
        np.asarray([colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1) for s in random_states]) * 255
    ).astype(np.uint8)
    _imwrite(output_path, unique_colors[indices].reshape((H, W, 3)))


def visualize_bw(input_path: Path, output_path: Path) -> None:
    data = load_single_channel(input_path)
    _imwrite(output_path, data)


# ---------------------------------------------------------------------------
# GT-type → visualizer mapping (uses cvdpack gt_type strings, not ExportType)
# ---------------------------------------------------------------------------

VISUALIZATION_FUNCS: dict[str, Callable] = {
    "depth": visualize_depth,
    "optical-flow": visualize_flow,
    "surface-normal": visualize_normals,
    "semantic-segmentation": visualize_seg_mask,
    "material-segmentation": visualize_seg_mask,
    "instance-segmentation": visualize_uniq_inst,
    "flow-mask-matched": visualize_bw,
    "flow-mask-cycle-consistency": visualize_bw,
    "occlusion": visualize_bw,
}


# ---------------------------------------------------------------------------
# Template derivation
# ---------------------------------------------------------------------------


def derive_vis_png_template(original: str) -> str:
    """Prepend vis_ to the filename and change extension to .png.

    e.g. '{scene}/{cam}/depth_{frame:04d}.npy' -> '{scene}/{cam}/vis_depth_{frame:04d}.png'
    """
    p = Path(original)
    new_name = "vis_" + re.sub(r"\.[^.{]+$", ".png", p.name)
    return str(p.parent / new_name)


def derive_vis_mkv_template(packed: str) -> Optional[str]:
    """Prepend vis_ to the MKV filename.

    e.g. '{scene}/depth-{cam}.mkv' -> '{scene}/vis_depth-{cam}.mkv'
    Returns None if packed is not an MKV.
    """
    if not packed.endswith(".mkv"):
        return None
    p = Path(packed)
    return str(p.parent / ("vis_" + p.name))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def visualize_frameset(input_template: Path, output_template: Path, gt_type: str) -> None:
    """Visualize all frames matching input_template, writing PNGs to output_template."""
    if gt_type not in VISUALIZATION_FUNCS:
        raise ValueError(f"No visualizer registered for {gt_type=}. Known types: {list(VISUALIZATION_FUNCS)}")

    frame_matches = sorted(util.match_template_paths(input_template), key=lambda x: x[1])
    if not frame_matches:
        raise ValueError(f"No frames found matching {input_template}")

    output_template.parent.mkdir(parents=True, exist_ok=True)

    # Compute global stats for types that need consistent normalization across frames
    kwargs = {}
    if gt_type in ("depth", "optical-flow") and len(frame_matches) > 1:
        vmin, vmax = float("inf"), float("-inf")
        for _, frame_path in frame_matches:
            data = load_data(frame_path)
            if gt_type == "depth":
                valid = (data > 1e-3) & (data < 1e4)
                if valid.any():
                    vmin = min(vmin, float(data[valid].min()))
                    vmax = max(vmax, float(data[valid].max()))
            else:
                vmin = min(vmin, float(data[..., :2].min()))
                vmax = max(vmax, float(data[..., :2].max()))
        kwargs = {"vmin": vmin, "vmax": vmax}

    func = VISUALIZATION_FUNCS[gt_type]
    for frame_info, frame_path in frame_matches:
        out_path = util.format_template(output_template, frame_info)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        func(frame_path, out_path, **kwargs)
        logger.debug(f"Visualized {frame_path} -> {out_path}")
