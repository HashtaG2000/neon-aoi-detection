"""
Optional AOI mask authoring with pupil-labs/aois_module.

This helper is intentionally separate from the Qt review app. It lets you use the
heavy GroundingSAM stack once to create precise reference masks, while the normal
analysis path only needs OpenCV to consume the resulting PNG masks.

Example:
    python App/tools/generate_aoi_masks.py screen_ref.png --surface Screen --search "avatar. progress bar. points bar."
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import cv2
import numpy as np

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import CONFIG_DIR


def _safe_name(label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip()).strip("_")
    return name or "AOI"


def _load_generator():
    try:
        from pupil_labs.aois_module._AOIs import AOI_Generator
    except Exception as exc:
        raise RuntimeError(
            "pupil-labs-aois-module is not installed or its ML dependencies are missing. "
            "Install it in a separate environment if possible: "
            "pip install pupil-labs-aois-module torch torchvision"
        ) from exc
    return AOI_Generator()


def generate_masks(image_path: pathlib.Path, surface: str, search: str, output_dir: pathlib.Path) -> pathlib.Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "aoi_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    generator = _load_generator()
    scaled = generator.scale_img(image)
    labels, boxes = generator.predict_dino(scaled, search)
    aois, _color_masks = generator.predict_sam(scaled, labels, boxes)

    regions = []
    for idx, row in aois.iterrows():
        label = _safe_name(str(row.get("label", f"AOI_{idx}")))
        mask = np.asarray(row["segmentation"], dtype=np.uint8) * 255
        mask_path = mask_dir / f"{label}.png"
        cv2.imwrite(str(mask_path), mask)
        regions.append(
            {
                "name": label,
                "surface": surface,
                "path": str(mask_path.relative_to(output_dir)).replace("\\", "/"),
                "priority": 200 - idx,
            }
        )

    config_path = output_dir / "aoi_masks.json"
    payload = {"version": 1, "regions": regions}
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AOI mask config from a reference image.")
    parser.add_argument("image", type=pathlib.Path, help="Reference image for the surface.")
    parser.add_argument("--surface", default="Screen", help="Marker-mapper surface name to attach masks to.")
    parser.add_argument("--search", required=True, help="GroundingDINO prompt, e.g. 'avatar. progress bar.'")
    parser.add_argument("--output-dir", type=pathlib.Path, default=CONFIG_DIR, help="Folder for aoi_masks.json and masks.")
    args = parser.parse_args()

    try:
        config_path = generate_masks(args.image, args.surface, args.search, args.output_dir)
    except Exception as exc:
        print(f"Failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
