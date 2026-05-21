import cv2
import pupil_apriltags
import os
import pathlib
import sys

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import IMAGES_DIR

detector = pupil_apriltags.Detector(families='tag36h11')
for f in os.listdir(IMAGES_DIR):
    if f.endswith('.jpeg'):
        img = cv2.imread(str(IMAGES_DIR / f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        ids = [d.tag_id for d in tags]
        print(f'{f}: {ids}')
