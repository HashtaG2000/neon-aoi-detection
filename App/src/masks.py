from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


DEFAULT_CONFIG_NAME = "aoi_masks.json"


@dataclass(frozen=True)
class AoiRegion:
    name: str
    surface: str
    kind: str
    priority: int = 0
    bounds: tuple[float, float, float, float] | None = None
    points: np.ndarray | None = None
    mask_path: pathlib.Path | None = None
    _mask: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def contains(self, surface_xy: np.ndarray) -> bool:
        if surface_xy is None or len(surface_xy) < 2:
            return False
        x, y = float(surface_xy[0]), float(surface_xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False

        if self.kind == "bbox" and self.bounds is not None:
            u1, u2, v1, v2 = self.bounds
            return u1 <= x <= u2 and v1 <= y <= v2

        if self.kind == "polygon" and self.points is not None and len(self.points) >= 3:
            return cv2.pointPolygonTest(
                self.points.reshape(-1, 1, 2).astype(np.float32),
                (x, y),
                False,
            ) >= 0

        if self.kind == "mask":
            mask = self.mask
            if mask is None:
                return False
            h, w = mask.shape[:2]
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                return False
            px = min(w - 1, max(0, int(round(x * (w - 1)))))
            py = min(h - 1, max(0, int(round(y * (h - 1)))))
            return bool(mask[py, px] > 0)

        return False

    @property
    def mask(self) -> np.ndarray | None:
        loaded = object.__getattribute__(self, "_mask")
        if loaded is not None:
            return loaded
        if self.mask_path is None or not self.mask_path.exists():
            return None
        img = cv2.imread(str(self.mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            mask = img[:, :, 3]
        elif img.ndim == 3:
            mask = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            mask = img
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        object.__setattr__(self, "_mask", binary)
        return binary

    def normalized_outline(self, max_points: int = 160) -> np.ndarray | None:
        if self.kind == "bbox" and self.bounds is not None:
            u1, u2, v1, v2 = self.bounds
            return np.array(
                [[u1, v1], [u2, v1], [u2, v2], [u1, v2]],
                dtype=np.float32,
            )

        if self.kind == "polygon" and self.points is not None:
            return self.points.astype(np.float32)

        if self.kind != "mask":
            return None
        mask = self.mask
        if mask is None:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
        contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(contour) > max_points:
            step = int(np.ceil(len(contour) / max_points))
            contour = contour[::step]
        h, w = mask.shape[:2]
        denom_x = max(1, w - 1)
        denom_y = max(1, h - 1)
        return np.column_stack((contour[:, 0] / denom_x, contour[:, 1] / denom_y)).astype(np.float32)


class AoiMaskConfig:
    def __init__(self, regions: list[AoiRegion], config_path: pathlib.Path | None = None):
        self.config_path = config_path
        self.regions = sorted(regions, key=lambda region: region.priority, reverse=True)
        self.by_surface: dict[str, list[AoiRegion]] = {}
        for region in self.regions:
            self.by_surface.setdefault(region.surface, []).append(region)

    @property
    def names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for region in self.regions:
            if region.name not in seen:
                seen.add(region.name)
                names.append(region.name)
        return names

    def for_surface(self, surface_name: str) -> list[AoiRegion]:
        return self.by_surface.get(surface_name, [])

    @property
    def has_custom_config(self) -> bool:
        return self.config_path is not None and self.config_path.exists()


def load_aoi_mask_config(
    app_dir: pathlib.Path,
    fallback_screen_boxes: dict[str, tuple[float, float, float, float]] | None = None,
) -> AoiMaskConfig:
    config_path = app_dir / DEFAULT_CONFIG_NAME
    if config_path.exists():
        return AoiMaskConfig(_load_regions(config_path), config_path)

    fallback_regions = [
        AoiRegion(
            name=name,
            surface="Screen",
            kind="bbox",
            priority=10,
            bounds=(float(u1), float(u2), float(v1), float(v2)),
        )
        for name, (u1, u2, v1, v2) in (fallback_screen_boxes or {}).items()
    ]
    return AoiMaskConfig(fallback_regions)


def _load_regions(config_path: pathlib.Path) -> list[AoiRegion]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    regions: list[AoiRegion] = []

    if isinstance(payload, dict) and isinstance(payload.get("surfaces"), dict):
        for surface_name, surface_payload in payload["surfaces"].items():
            masks = surface_payload.get("masks", []) if isinstance(surface_payload, dict) else []
            regions.extend(_parse_region_items(masks, str(surface_name), base_dir))
    elif isinstance(payload, dict) and isinstance(payload.get("regions"), list):
        regions.extend(_parse_region_items(payload["regions"], None, base_dir))
    elif isinstance(payload, list):
        regions.extend(_parse_region_items(payload, None, base_dir))

    return regions


def _parse_region_items(
    items: list[Any],
    default_surface: str | None,
    base_dir: pathlib.Path,
) -> list[AoiRegion]:
    regions: list[AoiRegion] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("label") or "").strip()
        surface_name = str(item.get("surface") or default_surface or "").strip()
        if not name or not surface_name:
            continue
        priority = int(item.get("priority", 100 - idx))

        if "bounds" in item:
            bounds = _parse_bounds(item["bounds"])
            if bounds is not None:
                regions.append(AoiRegion(name, surface_name, "bbox", priority, bounds=bounds))
            continue

        if "points" in item:
            points = np.asarray(item["points"], dtype=np.float32)
            if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3:
                regions.append(AoiRegion(name, surface_name, "polygon", priority, points=points))
            continue

        mask_file = item.get("path") or item.get("file") or item.get("mask")
        if mask_file:
            mask_path = pathlib.Path(str(mask_file))
            if not mask_path.is_absolute():
                mask_path = base_dir / mask_path
            regions.append(AoiRegion(name, surface_name, "mask", priority, mask_path=mask_path))

    return regions


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        try:
            return (
                float(value["u_min"]),
                float(value["u_max"]),
                float(value["v_min"]),
                float(value["v_max"]),
            )
        except KeyError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        u1, u2, v1, v2 = value
        return float(u1), float(u2), float(v1), float(v2)
    return None
