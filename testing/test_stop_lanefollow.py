#!/usr/bin/env python3
import asyncio
import time
import websockets
import json
import base64
import numpy as np
import cv2
import sys

BOT = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
WS_URL = f"ws://{BOT}.local:9001/ros_api"
CAMERA_TOPIC = f"/{BOT}/camera_node/image/compressed"

# Topics & services
MANUAL_CMD_TOPIC = f"/{BOT}/car_cmd_switch_node/cmd"
LANE_FOLLOW_TOPIC = f"/{BOT}/lane_controller_node/car_cmd"
LANE_FOLLOW_SWITCH = f"/{BOT}/lane_controller_node/switch"

# Movement parameters
SPEED = 0.5
TURN = 0.7

# Toggle
lane_following = True
is_stopped = False

# Store latest lane-following command
latest_lane_cmd = {"v": 0.0, "omega": 0.0}

# -------------------------------
# Send velocity over ROS bridge
# -------------------------------
async def send_cmd(ws, topic, v, omega):
    msg = {
        "op": "publish",
        "topic": topic,
        "msg": {"v": v, "omega": omega},
    }
    await ws.send(json.dumps(msg))
# -------------------------------
# Call lane following switch service
# -------------------------------
async def switch_lane_follow(ws, enable: bool):
    msg = {
        "op": "call_service",
        "service": LANE_FOLLOW_SWITCH,
        "args": {"data": enable},
    }
    await ws.send(json.dumps(msg))

# -------------------------------
# Lane following subscriber
# -------------------------------
async def lane_follow_listener():
    global latest_lane_cmd
    async with websockets.connect(WS_URL) as ws:
        sub_msg = {
            "op": "subscribe",
            "topic": LANE_FOLLOW_TOPIC
        }
        await ws.send(json.dumps(sub_msg))

        while True:
            try:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")
                msg = json.loads(raw)
                if msg.get("topic") == LANE_FOLLOW_TOPIC:
                    lane_msg = msg.get("msg", {})
                    latest_lane_cmd["v"] = lane_msg.get("v", 0.0)
                    latest_lane_cmd["omega"] = lane_msg.get("omega", 0.0)
                    print(f"Lane follow cmd: v={latest_lane_cmd['v']}, omega={latest_lane_cmd['omega']}")
            except Exception:
                await asyncio.sleep(0.01)

def observingBox(x_pos, y_pos, mask):
    for x in range(x_pos, x_pos + 100):
        for y in range(y_pos, y_pos + 28):
            if mask[y, x] != 0:
                return True
    return False

async def main():
    global is_stopped
    global lane_following
    async with websockets.connect(WS_URL) as ws:
        asyncio.create_task(lane_follow_listener())
        await ws.send(json.dumps({"op": "subscribe", "topic": CAMERA_TOPIC}))
        last_send = time.time()
        last_stop = 0
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("topic") == CAMERA_TOPIC:
                data = msg["msg"]["data"]
                arr = np.frombuffer(base64.b64decode(data), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    
                    mask1 = cv2.inRange(hsv, lower_red, upper_red)

                    # result = cv2.bitwise_and(img, img, mask=mask1)

                    #checking for red recttangle
                    cv2.rectangle(img,(304,400),(404,428),(0,255,0),3)

                    if not is_stopped:
                        if observingBox(304, 400, mask1):
                            is_stopped = True
                            lane_following = not lane_following
                            print("Lane following =", lane_following)
                            await switch_lane_follow(ws, lane_following)
                            last_stop = time.time()
                        else:
                            print("Moving forward")
                    else:
                        print("Stopped at crossroad")
                        if time.time() - last_send > 0.05:
                            if time.time() - last_stop < 2:
                                await send_cmd(ws, MANUAL_CMD_TOPIC, 0, 0)
                                # last_stop = time.time()
                            elif time.time() - last_stop < 2 + 1.5:
                                #LEFT turn - 4 seconds added
                                # -----------------------------------
                                #await send_cmd(ws, MANUAL_CMD_TOPIC, 0.1, 1.3)
                                #RIGHT turn - 3 seconds added
                                # -----------------------------------
                                # await send_cmd(ws, MANUAL_CMD_TOPIC, 0.1, -1.75)
                                #U turn - mirror right for 1.5 - 3 seconds added
                                # -----------------------------------
                                await send_cmd(ws, MANUAL_CMD_TOPIC, 0.1, 1.75)
                                # await asyncio.sleep(0.01)
                                # await send_cmd(ws, MANUAL_CMD_TOPIC, SPEED , TURN + 0.2)
                            elif time.time() - last_stop < 2 + 3:
                                #U turn - mirror right for 1.5 - 3 seconds added
                                # -----------------------------------
                                await send_cmd(ws, MANUAL_CMD_TOPIC, 0.1, 2)
                            elif time.time() - last_stop < 7:
                                is_stopped = False
                                lane_following = not lane_following
                                print("Lane following =", lane_following)
                                await switch_lane_follow(ws, lane_following)
                            last_send = time.time()

                        await asyncio.sleep(0.01)
                        # if time.time() - last_stop > 2.3:
                        #     is_stopped = False
                        #TODO: implement wait and turn logic here

                    cv2.imshow("frame", img)
                    # cv2.imshow("mask", mask1)
                    # cv2.imshow("result", result)
                    if cv2.waitKey(1) & 0xFF == 27:
                        await send_cmd(ws, MANUAL_CMD_TOPIC, 0, 0)
                        break

asyncio.run(main())
cv2.destroyAllWindows()
