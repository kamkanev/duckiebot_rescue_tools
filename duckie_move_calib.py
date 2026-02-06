import asyncio
import websockets
import pygame
import json
import sys

# -----------------------------
# Config
# -----------------------------
BOT = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
WS_URL = f"ws://{BOT}.local:9001/ros_api"

BASE_SPEED = 0.5
ANGULAR_SPEED = 0.5
GAIN_STEP = 0.01  # how much to change per key press

TOPIC = f"/{BOT}/car_cmd_switch_node/cmd"

# -----------------------------
# Pygame setup
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((500, 400))
pygame.display.set_caption("Duckiebot Teleop + Motor Calibration")
font = pygame.font.SysFont("Arial", 24)

# motor gains
gain_left = 1.0
gain_right = 1.0
trim_left = 0.0
trim_right = 0.0

def draw(action, gain_left, gain_right, trim_left, trim_right):
    screen.fill((30, 30, 30))
    label_color = (255, 255, 255)

    # Show action
    label = font.render(f"Action: {action}", True, label_color)
    screen.blit(label, (20, 20))

    # Show gains
    glabel = font.render(f"Left Gain: {gain_left:.2f}", True, label_color)
    rlabel = font.render(f"Right Gain: {gain_right:.2f}", True, label_color)
    screen.blit(glabel, (20, 60))
    screen.blit(rlabel, (20, 100))

    # Show trims
    tlabel_left = font.render(f"Left Trim: {trim_left:.2f}", True, label_color)
    tlabel_right = font.render(f"Right Trim: {trim_right:.2f}", True, label_color)
    screen.blit(tlabel_left, (20, 140))
    screen.blit(tlabel_right, (20, 180))

    # Show instructions
    instructions = [
        "[W/A/S/D] Move",
        "[Q/E] Increase/Decrease Left motor gain",
        "[Z/C] Increase/Decrease Right motor gain",
        "[R/F] Increase/Decrease Left motor trim",
        "[T/G] Increase/Decrease Right motor trim",
        "[ESC] Stop & exit"
    ]
    y = 260
    for instr in instructions:
        txt = font.render(instr, True, label_color)
        screen.blit(txt, (20, y))
        y += 30

    pygame.display.flip()

# -----------------------------
# Send velocity over WebSocket
# -----------------------------
async def send_velocity(ws, v, omega, gain_left, gain_right, trim_left, trim_right):
    # Apply gains by scaling left/right wheels for differential drive
    # Assuming differential drive: v_l = v - omega, v_r = v + omega
    v_l = (v - omega) * gain_left + trim_left
    v_r = (v + omega) * gain_right + trim_right
    
    v_avg = (v_l + v_r) / 2
    omega_adj = (v_r - v_l) / 2

    print(f"Sending velocities => v: {v_avg:.2f}, omega: {omega_adj:.2f}")
    msg = {
        "op": "publish",
        "topic": TOPIC,
        "msg": {"v": v_avg, "omega": omega_adj}
    }
    await ws.send(json.dumps(msg))

# -----------------------------
# Main loop
# -----------------------------
async def main():
    global gain_left, gain_right, trim_left, trim_right
    try:
        async with websockets.connect(WS_URL) as ws:
            clock = pygame.time.Clock()
            running = True
            while running:
                v, omega = 0.0, 0.0
                action = "Stop"

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                keys = pygame.key.get_pressed()
                # movement
                if keys[pygame.K_w]:
                    v = BASE_SPEED
                    action = "Forward"
                elif keys[pygame.K_s]:
                    v = -BASE_SPEED
                    action = "Backward"
                if keys[pygame.K_a]:
                    omega = ANGULAR_SPEED
                    action = "Left"
                elif keys[pygame.K_d]:
                    omega = -ANGULAR_SPEED
                    action = "Right"

                # motor gain adjustment
                if keys[pygame.K_q]:  # increase left gain
                    gain_left += GAIN_STEP
                if keys[pygame.K_e]:  # decrease left gain
                    gain_left = max(0, gain_left - GAIN_STEP)
                if keys[pygame.K_z]:  # increase right gain
                    gain_right += GAIN_STEP
                if keys[pygame.K_c]:  # decrease right gain
                    gain_right = max(0, gain_right - GAIN_STEP)

                if keys[pygame.K_r]:  # increase left trim
                    trim_left += 0.01
                if keys[pygame.K_f]:  # decrease left trim
                    trim_left -= 0.01
                if keys[pygame.K_t]:  # increase right trim
                    trim_right += 0.01
                if keys[pygame.K_g]:  # decrease right trim
                    trim_right -= 0.01

                # ESC to exit
                if keys[pygame.K_ESCAPE]:
                    running = False

                draw(action, gain_left, gain_right, trim_left, trim_right)
                await send_velocity(ws, v, omega, gain_left, gain_right, trim_left, trim_right)
                clock.tick(20)

            # stop robot on exit
            await send_velocity(ws, 0.0, 0.0, gain_left, gain_right, trim_left, trim_right)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
