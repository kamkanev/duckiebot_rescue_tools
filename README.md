# Duckietown Rescue Tools

A collection of teleoperation and calibration utilities for Duckiebot robots.

## Scripts Overview

### 1. `duckie_move_calib.py` - Motor Calibration & Teleoperation
Calibrate motor gains and trim values while controlling the robot in real-time.

**Features:**
- WebSocket-based communication via rosbridge
- Adjust left/right motor gains independently
- Fine-tune motor trim values
- Live on-screen feedback of current settings

**Usage:**
```bash
python3 duckie_move_calib.py <bot_name>
```

**Example:**
```bash
python3 duckie_move_calib.py entebot208
```

**Controls:**
- **W/A/S/D** - Move the robot (Forward/Left/Backward/Right)
- **Q** - Increase left motor gain
- **E** - Decrease left motor gain
- **Z** - Increase right motor gain
- **C** - Decrease right motor gain
- **R** - Increase left motor trim
- **F** - Decrease left motor trim
- **T** - Increase right motor trim
- **G** - Decrease right motor trim
- **ESC** - Stop and exit

**Note:** The gains and trim adjustments are applied to the differential drive calculations. Typical starting gain is 1.0 and trim is 0.0.


---

### 3. `test_camera.py` - Quick Camera Connectivity Test
Test if the robot's camera feed is accessible before running the full camera GUI.

**Usage:**
```bash
python3 test_camera.py <bot_name>
```

**Example:**
```bash
python3 test_camera.py entebot208
```

**Output:**
- Attempts to connect to the robot's camera WebSocket
- Reports success or connection errors
- Useful for troubleshooting network and API issues

---

### 4. `duckie_camera_gui.py` - Camera Feed & Image Capture
View the robot's camera feed in real-time and capture calibration images.

**Features:**
- Live video stream from the robot
- Capture images for calibration
- On-screen instructions overlay
- Feedback message when image is saved
- Automatic retry logic for connection failures
- Support for both bot name and IP address

This ensures you start with a fresh set of calibration images.

**Usage:**
```bash
python3 duckie_camera_gui.py <bot_name>
```

**Examples:**
```bash
python3 duckie_camera_gui.py entebot208
```

**Controls:**
- **C** - Capture the current camera frame and save to `~/duckietown/calibration/`
- **ESC** - Exit the camera view

**Output:**
Captured images are saved as `calib_000.png`, `calib_001.png`, etc. in the calibration folder.

---

### 5. `calibrate_intrinsics.py` - Camera Intrinsics Calibration
Calibrate camera intrinsic parameters (focal length, principal point, distortion coefficients) using checkerboard images.

**Prerequisites - IMPORTANT:**
Before running intrinsics calibration, **clean up completely**:
```bash
# Remove old calibration folder and all files
rm -rf ~/duckietown/calibration_old
rm -rf ~/duckietown/calibration
mkdir -p ~/duckietown/calibration
```

**Workflow:**

1. **Capture 20-30 checkerboard images** using `duckie_camera_gui.py`:
   ```bash
   python3 duckie_camera_gui.py entebot208
   ```
   
2. **Position the Duckietown checkerboard vertically** in front of the robot's camera:
   - Ensure good lighting (avoid shadows and reflections)
   - Capture images from **different angles and distances**:
     - Center position (multiple distances)
     - Left side at an angle
     - Right side at an angle
     - Top-down angle
     - Low angle (camera looking up)
     - Rotated checkerboard positions
   
   - Aim for **20-30 total images** with the checkerboard clearly visible
   - **Example:** The checkerboard should fill 50-70% of the frame for best results

3. **Run the intrinsics calibration script**:
   ```bash
   python3 calibrate_intrinsics.py
   ```

**Features:**
- Detects checkerboard patterns automatically in all captured images
- Computes camera matrix (focal length, principal point)
- Calculates distortion coefficients (k1, k2, p1, p2, k3)
- Generates `camera_intrinsics/<bot_name>.yaml` with all calibration results
- Displays reprojection error for quality assessment
- Shows which images were successfully processed

**Output:**
- `camera_intrinsics/<bot_name>.yaml` - Camera intrinsic parameters (saved in `~/duckietown/calibration/`)
- Console output showing:
  - Number of images processed successfully
  - Camera matrix (3x3)
  - Distortion coefficients
  - Mean reprojection error (lower is better, <1.0 is good)

---

### 6. `calibrate_extrinsics.py` - Camera Extrinsics Calibration
Calibrate camera extrinsic parameters to determine the camera's position and orientation relative to the robot chassis.

**Prerequisites - IMPORTANT:**
Before running extrinsics calibration:
1. **Camera intrinsics MUST be calibrated first** (run `calibrate_intrinsics.py`)
2. **Clean the calibration folder**:
   ```bash
   # Remove old files but keep intrinsics.yaml
   rm -f ~/duckietown/calibration/calib_*.png
   rm -rf ~/duckietown/calibration_old
   ```

3. Ensure `<bot_name>.yaml` exists in the calibration_intrinsics folder

**Workflow:**

1. **Capture 20-30 extrinsics calibration images** using `duckie_camera_gui.py`:
   ```bash
   python3 duckie_camera_gui.py entebot208
   ```

2. **Position the checkerboard vertically** and capture from **different robot positions and angles**:
   - Robot close to checkerboard
   - Robot far from checkerboard (various distances)
   - Robot at left angle to checkerboard
   - Robot at right angle to checkerboard
   - Robot rotated at different angles
   
   - Capture **20-30 images minimum** for robust calibration
   - Ensure checkerboard is fully visible in each image

3. **Run the extrinsics calibration script**:
   ```bash
   python3 calibrate_extrinsics.py
   ```
   The script will:
   - Load the `<bot_name>.yaml` from previous calibration
   - Process all captured images
   - Compute camera-to-robot transformation

**Features:**
- Uses intrinsic parameters from `camera_intrinsics/<bot_name>.yaml`
- Detects checkerboard patterns to estimate camera pose
- Computes rotation matrix (3x3) and translation vector (3x1)
- Generates `camera_extrinsics/<bot_name>.yaml` with camera-to-robot transformation
- Validates calibration with reprojection tests
- Provides diagnostic information for quality assessment

**Output:**
- `camera_extrinsics/<bot_name>.yaml` - Camera extrinsic parameters (saved in `~/duckietown/calibration/`)
- Console output showing:
  - Rotation matrix (3x3) - camera orientation
  - Translation vector (3x1) - camera position (in meters)
  - Reprojection error
  - Number of successfully processed images
  - Camera pose relative to robot base

---

## Full Calibration Workflow

**Complete step-by-step process:**

```bash
# Step 1: Clean everything
rm -rf ~/duckietown/calibration_old
rm -rf ~/duckietown/calibration
mkdir -p ~/duckietown/calibration

# Step 2: Capture intrinsics images (20-30 images with checkerboard at different angles/distances)
python3 duckie_camera_gui.py entebot208

# Step 3: Run intrinsics calibration
python3 calibrate_intrinsics.py
# → Generates intrinsics.yaml

# Step 4: Clear only the PNG files (keep intrinsics.yaml)
rm -f ~/duckietown/calibration/calib_*.png

# Step 5: Capture extrinsics images (20-30 images with robot at different positions)
python3 duckie_camera_gui.py entebot208

# Step 6: Run extrinsics calibration
python3 calibrate_extrinsics.py
# → Generates extrinsics.yaml

# Step 7: Verify calibration (optional)
python3 duckie_teleop_gui.py entebot208  # Test robot movement
```

---

## Installation & Setup

### Requirements
- Python 3.10+
- pygame
- numpy
- opencv-python (cv2)
- websockets
- requests

### Install Dependencies
```bash
pip3 install pygame numpy opencv-python websockets requests
```

Or using the virtual environment:
```bash
source duckiebot-env/bin/activate
pip3 install pygame numpy opencv-python websockets requests
```

### Network Configuration
Ensure your machine can reach the Duckiebot:
- **mDNS (Recommended):** Use `<bot_name>.local` (e.g., `entebot208.local`)
- **Direct IP:** Use the robot's IP address (e.g., `172.21.37.170`)

Test connectivity:
```bash
ping entebot208.local
# or
ping 172.21.37.170
```

---

## Workflow: Camera Calibration & Motor Tuning

### Step 1: Capture Calibration Images
Clean the calibration folder, then run the camera GUI:
```bash
rm -rf ~/duckietown/calibration
mkdir -p ~/duckietown/calibration
python3 test_camera.py entebot208  # Test first
python3 duckie_camera_gui.py entebot208  # Capture images
```

Press **C** multiple times to capture different angles and lighting conditions.

### Step 2: Calibrate Motor Gains
After capturing images, run the motor calibration tool:
```bash
python3 duckie_move_calib.py entebot208
```

Adjust gains and trim values while observing the robot's movement:
- Start with small adjustments (±0.01 gain, ±0.005 trim)
- Test forward movement to check gain balance
- Adjust trim if the robot drifts left or right at zero omega

### Step 3: Verify with Teleop
Once calibration is complete, test with the teleop GUI:
```bash
python3 duckie_teleop_gui.py entebot208
```

Verify the robot moves straight and responds smoothly to all commands.

---

## Troubleshooting

### Connection Issues
- **"Temporary failure in name resolution"** → Use IP address instead of bot name
- **"I am not Web server, but a WebSocket Endpoint"** → The endpoint expects WebSocket; the scripts handle this automatically
- **Retry messages in console** → Robot may be booting or busy; wait and try again

### Camera Feed Not Showing
1. Run `test_camera.py` first to diagnose
2. Ensure the robot's camera node is running
3. Check network connectivity with `ping`
4. Verify the correct bot name or IP address

### Motor Not Responding
1. Confirm the robot is powered on and connected
2. Check that the correct WebSocket topic is being used (default: `/{BOT}/car_cmd_switch_node/cmd`)
3. Test with `duckie_teleop_gui.py` first; it has built-in fallback logic

### Images Not Being Saved
1. Ensure `~/duckietown/calibration/` directory exists and is writable
2. Check disk space: `df -h`
3. Verify file permissions: `ls -la ~/duckietown/`

---

## File Structure
```
~/duckietown/
├── duckie_teleop_gui.py      # Main teleop with arrow GUI
├── duckie_move_calib.py       # Motor calibration tool
├── duckie_camera_gui.py       # Camera feed & image capture
├── test_camera.py             # Quick camera test
├── calibration/               # Captured calibration images
│   ├── calib_000.png
│   ├── calib_001.png
│   └── ...
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## Notes

- **WebSocket Protocol:** All scripts communicate with the robot via WebSocket on port 9001
- **ROS Bridge:** Camera and motor commands use rosbridge-compatible JSON messages
- **Thread Safety:** Camera GUI and motor calibration use async/await for non-blocking I/O
- **Calibration Storage:** Motor gains/trim are not automatically saved; document them manually

---

## License
See LICENSE file for details.

---

## Support
For issues or questions, check the console output for detailed error messages and connection diagnostics.
