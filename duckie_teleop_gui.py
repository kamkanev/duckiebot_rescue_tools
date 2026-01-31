#!/usr/bin/env python3
import asyncio
import websockets
import json
import pygame
import sys
import time

BOT = "entebot208"
WS = f"ws://{BOT}.local:9001/ros_api"

# Topics & services
MANUAL_CMD_TOPIC = f"/{BOT}/car_cmd_switch_node/cmd"
LANE_FOLLOW_TOPIC = f"/{BOT}/lane_controller_node/car_cmd"
LANE_FOLLOW_SWITCH = f"/{BOT}/lane_controller_node/switch"

# Movement parameters
SPEED = 0.3
TURN = 0.7

# Toggle
lane_following = False

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
async def lane_follow_listener(ws):
    global latest_lane_cmd
    sub_msg = {
        "op": "subscribe",
        "topic": LANE_FOLLOW_TOPIC
    }
    await ws.send(json.dumps(sub_msg))

    while True:
        try:
            raw = await ws.recv()
            print(raw)
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

# -------------------------------
# Main teleop loop
# -------------------------------
async def teleop():
    global lane_following
    pygame.init()
    screen = pygame.display.set_mode((600, 300))
    pygame.display.set_caption("Duckiebot Teleop with Lane Follow Toggle")
    font = pygame.font.SysFont("Arial", 28)

    async with websockets.connect(WS) as ws:
        # Start lane following listener in background
        asyncio.create_task(lane_follow_listener(ws))
        last_send = time.time()

        while True:
            screen.fill((0, 0, 0))

            mode_text = "MODE: LANE FOLLOWING" if lane_following else "MODE: MANUAL"
            text_surface = font.render(mode_text, True, (255, 255, 0))
            screen.blit(text_surface, (20, 20))

            controls = [
                "↑ Forward",
                "↓ Backward",
                "← Turn Left",
                "→ Turn Right",
                "F Toggle Lane Following",
                "ESC Quit",
            ]
            for i, t in enumerate(controls):
                s = font.render(t, True, (200, 200, 200))
                screen.blit(s, (20, 60 + i * 25))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

                    if event.key == pygame.K_f:
                        lane_following = not lane_following
                        print("Lane following =", lane_following)
                        await switch_lane_follow(ws, lane_following)
                        

            # Decide which commands to send
            if lane_following:
                pass
                # v = SPEED
                # omega = 0.0
            else:
                keys = pygame.key.get_pressed()
                v, omega = 0.0, 0.0
                if keys[pygame.K_UP]:
                    v = SPEED
                if keys[pygame.K_DOWN]:
                    v = -SPEED
                if keys[pygame.K_LEFT]:
                    omega = TURN
                if keys[pygame.K_RIGHT]:
                    omega = -TURN

                # Avoid spamming ROS
                if time.time() - last_send > 0.05:
                    await send_cmd(ws, MANUAL_CMD_TOPIC, v, omega)
                    last_send = time.time()

                await asyncio.sleep(0.01)

# -------------------------------
# Run main
# -------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(teleop())
    except KeyboardInterrupt:
        print("Exiting.")
