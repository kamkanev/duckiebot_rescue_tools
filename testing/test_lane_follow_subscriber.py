#!/usr/bin/env python3
"""Subscribe to lane following topic over ROS WebSocket API and print values."""

import asyncio
import websockets
import json
import sys

BOT = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
WS_URL = f"ws://{BOT}.local:9001/ros_api"
LANE_FOLLOW_TOPIC = f"/{BOT}/lane_controller_node/car_cmd"


async def listen_loop():
    backoff = 1
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print(f"Connected to {WS_URL}, subscribing to {LANE_FOLLOW_TOPIC}")
                await ws.send(json.dumps({"op": "subscribe", "topic": LANE_FOLLOW_TOPIC}))
                backoff = 1
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        print("Received non-JSON message:\n", raw)
                        continue

                    topic = msg.get("topic")
                    if topic != LANE_FOLLOW_TOPIC:
                        # ignore other topics, but print for debugging
                        # print(f"[{topic}] {msg.get('msg')}")
                        continue

                    payload = msg.get("msg", {})
                    # lane controller uses fields like 'v' and 'omega'
                    v = payload.get("v")
                    omega = payload.get("omega")
                    print(f"lane_cmd -> v={v}  omega={omega}  full={payload}")

        except Exception as e:
            print(f"Connection error: {e}; reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == '__main__':
    try:
        asyncio.run(listen_loop())
    except KeyboardInterrupt:
        print("Subscriber stopped by user")
