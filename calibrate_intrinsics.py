import cv2
import numpy as np
import yaml
import glob
import os
import sys

ROBOT_NAME = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
IMAGE_FOLDER = "./calibration"    # folder with checkerboard screenshots
OUTPUT_FILE = f"./camera_intrinsic/{ROBOT_NAME}.yaml"

# Duckietown DB21J Checkerboard
CHECKERBOARD = (7, 5)      # inner corners
SQUARE_SIZE = 0.031        # 31 mm printed square size



def find_intrinsics():
    obj_points = []   # 3D points
    img_points = []   # 2D points

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    images = glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
             glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))

    if len(images) == 0:
        print("No images found in:", IMAGE_FOLDER)
        sys.exit(1)

    print(f"Found {len(images)} images, processing…")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            print(f"✔ Checkerboard detected: {fname}")

            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            )

            obj_points.append(objp)
            img_points.append(corners)
        else:
            print(f"✖ No checkerboard in: {fname}")

    if len(obj_points) < 3:
        print("Need at least 3 valid images for calibration.")
        sys.exit(1)

    print("\nRunning calibration…")

    # Calibrate
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print("Calibration error:", ret)
    return K, D


def save_yaml(K, D):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    data = {
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": K.flatten().tolist()
        },
        "distortion_coefficients": {
            "rows": 1, "cols": len(D),
            "data": D.flatten().tolist()
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        yaml.dump(data, f)

    print("\n============================================")
    print("Intrinsic calibration saved to:")
    print(OUTPUT_FILE)
    print("============================================")


def main():
    K, D = find_intrinsics()
    save_yaml(K, D)


if __name__ == "__main__":
    main()
