#WORK IN PROGRESS
import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import os
from roboflow import Roboflow

# -----------------------------
# Config
# -----------------------------
BOT = "entebot208"
WS_URL = f"ws://{BOT}.local:9001/ros_api"
CAMERA_TOPIC = f"/{BOT}/camera_node/image/compressed"
CAPTURE_DIR = os.path.expanduser("~/duckietown/calibration")
os.makedirs(CAPTURE_DIR, exist_ok=True)

save_counter = len([n for n in os.listdir(CAPTURE_DIR) if n.lower().endswith((".png", ".jpg", ".jpeg"))])
save_feedback_time = 0.0
save_feedback_text = ""

# -----------------------------
# Roboflow Inference client
# -----------------------------
rf = Roboflow(api_key="0vxGwFAZFGqqkfpESMNe")
project = rf.workspace().project("duckietown-dswll")
model = project.version(3).model

# -----------------------------
# Detect if GUI is available
# -----------------------------
def gui_available():
    try:
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
        return True
    except cv2.error:
        return False

USE_GUI = gui_available()
if USE_GUI:
    print("[INFO] GUI detected: cv2.imshow() will be used.")
else:
    print("[INFO] No GUI detected: saving frames to /tmp/current_frame.jpg")

# -----------------------------
# Helper: Run inference
# -----------------------------
def run_inference(img):
    try:
        result = model.predict(img, confidence=40, overlap=30).json()
        print("[inference] Result:", result)
    except Exception as e:
        print("[inference] ERROR:", e)

# -----------------------------
# Main loop
# -----------------------------
async def main():
    global save_counter, save_feedback_time, save_feedback_text
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"op": "subscribe", "topic": CAMERA_TOPIC}))
        print(f"Subscribed to {CAMERA_TOPIC}. Press 'C' to capture, 'ESC' to exit.")

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("topic") == CAMERA_TOPIC:
                data = msg["msg"]["data"]
                arr = np.frombuffer(base64.b64decode(data), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Overlay info text
                    overlay_img = img.copy()
                    cv2.putText(overlay_img, "Press 'C' to capture image", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay_img, "Press 'ESC' to exit", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if save_feedback_time > 0 and (cv2.getTickCount() / cv2.getTickFrequency() - save_feedback_time) < 1.5:
                        cv2.putText(overlay_img, save_feedback_text, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Show GUI if available, else save image to disk
                    if USE_GUI:
                        cv2.imshow("Duckiebot Camera", overlay_img)
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        cv2.imwrite("/tmp/current_frame.jpg", overlay_img)
                        key = -1  # no key input in headless mode

                    # Run inference on the current frame
                    run_inference(img)

                    # Capture or exit
                    if key == 27:  # ESC
                        print("Exiting...")
                        break
                    elif key == ord('c') or key == ord('C'):
                        filename = os.path.join(CAPTURE_DIR, f"calib_{save_counter:03d}.png")
                        if cv2.imwrite(filename, img):
                            save_feedback_text = f"Saved {os.path.basename(filename)}"
                            save_feedback_time = cv2.getTickCount() / cv2.getTickFrequency()
                            print(f"[capture] saved {filename}")
                            save_counter += 1
                        else:
                            save_feedback_text = "Save failed"
                            save_feedback_time = cv2.getTickCount() / cv2.getTickFrequency()
                            print("[capture] ERROR: failed to save", filename)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
        if USE_GUI:
            cv2.destroyAllWindows()
