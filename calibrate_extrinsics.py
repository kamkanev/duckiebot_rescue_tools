#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import glob
import sys
import os


ROBOT_NAME = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
IMAGE_FOLDER = "./calibration"   # folder with screenshots
OUTPUT_FILE = f"./camera_extrinsic/{ROBOT_NAME}.yaml"

# Checkerboard specification (Duckietown DB21J)
CHECKERBOARD = (7, 5)     # internal corners
SQUARE_SIZE = 0.031       # 31mm printed squares

# Extrinsics board printed displacement:
# (from board origin to Duckiebot base_link origin)
# these values come from the DB21J board:
OFFSET_X = 0.160     # 160 mm forward
OFFSET_Y = -0.124    # -124 mm to the right
OFFSET_Z = 0.0       # camera sits above plane


def load_intrinsics():
    """Load intrinsics from camera_intrinsic/<robot>.yaml"""
    intr_file = f"./camera_intrinsic/{ROBOT_NAME}.yaml"
    print(f"Loading intrinsics from {intr_file}")

    with open(intr_file, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"])

    return K, D


def find_checkerboard_pose(image, K, D):
    """Find pose using checkerboard."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return None, None

    # 3D checkerboard points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # refine corners
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    # solvePnP
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
    if not ok:
        return None, None

    return rvec, tvec


def save_yaml(R, t):
    """Save YAML extrinsics file in Duckietown format."""
    data = {
        "camera_extrinsic": {
            "homography": {
                "rows": 3,
                "cols": 3,
                "data": R.flatten().tolist()
            },
            "translation": {
                "rows": 3,
                "cols": 1,
                "data": t.flatten().tolist()
            }
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        yaml.dump(data, f)

    print("===========================================")
    print("Extrinsic calibration saved to:")
    print(OUTPUT_FILE)
    print("===========================================")


def main():
    K, D = load_intrinsics()

    imgs = glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
           glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))

    if len(imgs) == 0:
        print("No images found in folder:", IMAGE_FOLDER)
        sys.exit(1)

    for img_path in imgs:
        print("Trying:", img_path)
        img = cv2.imread(img_path)

        rvec, tvec = find_checkerboard_pose(img, K, D)
        if rvec is None:
            print("No checkerboard detected")
            continue

        print("Checkerboard detected")

        # Convert rotation vector -> matrix
        R, _ = cv2.Rodrigues(rvec)

        # Apply Duckiebot printed board offsets:
        # transform board origin → robot base → camera
        tvec[0] += OFFSET_X
        tvec[1] += OFFSET_Y
        tvec[2] += OFFSET_Z

        save_yaml(R, tvec)
        print("Done.")
        return

    print("ERROR: No valid extrinsics image found.")


if __name__ == "__main__":
    main()
