#!/usr/bin/env python3
import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import os

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

                    # Show save feedback if recent
                    if save_feedback_time > 0 and (cv2.getTickCount() / cv2.getTickFrequency() - save_feedback_time) < 1.5:
                        cv2.putText(overlay_img, save_feedback_text, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Duckiebot Camera", overlay_img)
                    key = cv2.waitKey(1) & 0xFF

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

    cv2.destroyAllWindows()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
        cv2.destroyAllWindows()
