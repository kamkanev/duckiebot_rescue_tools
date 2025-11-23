#!/usr/bin/env python3
import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import sys

BOT = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
WS_URL = f"ws://{BOT}.local:9001/ros_api"
CAMERA_TOPIC = f"/{BOT}/camera_node/image/compressed"

async def main():
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"op": "subscribe", "topic": CAMERA_TOPIC}))
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("topic") == CAMERA_TOPIC:
                data = msg["msg"]["data"]
                arr = np.frombuffer(base64.b64decode(data), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow("frame", img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

asyncio.run(main())
cv2.destroyAllWindows()
