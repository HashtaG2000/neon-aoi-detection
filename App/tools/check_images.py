import cv2
import os
import pathlib
import sys

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import IMAGES_DIR

for f in os.listdir(IMAGES_DIR):
    if f.endswith('.jpeg'):
        img = cv2.imread(str(IMAGES_DIR / f))
        print(f'{f}: {img.shape}')
