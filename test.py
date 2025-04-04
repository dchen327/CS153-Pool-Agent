import pygetwindow as pgw
import mss
import cv2
import numpy as np
import subprocess
from time import sleep
from pathlib import Path
import pyautogui as pag

POWER_STICK_TOP = (112, 603)
POWER_STICK_BOTTOM = (112, 1200)

root = Path(__file__).parent
# Create screenshots directory if it doesn't exist
screenshot_dir = root / 'screenshots'
screenshot_dir.mkdir(exist_ok=True)

# Activate the window
try:
    window = pgw.getWindowsWithTitle('8Ball')[0]
    window.activate()
except:
    print("Window not found or couldn't be activated")
    # run scrcpy --no-audio --window-title=8Ball
    process = subprocess.Popen(['scrcpy', '--no-audio', '--fullscreen', '--window-title=8Ball'])
    sleep(4)  # Give scrcpy some time to open the window
    window = pgw.getWindowsWithTitle('8Ball')[0]
    window.activate()

# Get your actual monitor dimensions
# with mss.mss() as sct:
#     print("Available monitors: ", sct.monitors)
#     target_monitor = sct.monitors[1]  # Primary monitor
    
#     # Create the capture area using actual monitor dimensions
#     monitor = {
#         "top": target_monitor["top"], 
#         "left": target_monitor["left"], 
#         "width": target_monitor["width"], 
#         "height": target_monitor["height"]
#     }
    
#     # every 3s for 10 times
#     for i in range(10):
#         # Capture the screen
#         sct_img = sct.grab(monitor)
        
#         # Convert to CV2 format
#         img_cv2 = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        
#         # Write to file, save in screenshots folder
#         screenshot_path = screenshot_dir / f'screenshot_{i}.png'
#         cv2.imwrite(str(screenshot_path), img_cv2)
#         print(f"Screenshot {i} taken")
#         sleep(3)
def shoot_ball(coords, power=100):
    ''' Given (x, y) coordinates and power, click to aim and shoot the ball'''
    # Click to aim
    pag.click(coords[0], coords[1])
    sleep(0.3)  # Wait for the aim to settle
    # Drag power stick down to shoot
    pag.mouseDown(POWER_STICK_TOP[0], POWER_STICK_TOP[1])
    # Calculate the drag distance based on power
    drag_distance = int((POWER_STICK_BOTTOM[1] - POWER_STICK_TOP[1]) * (power / 100))
    # Drag down to shoot
    pag.moveTo(POWER_STICK_BOTTOM[0], POWER_STICK_TOP[1] + drag_distance, duration=0.25)
    # Release the mouse button
    pag.mouseUp()
    # Wait for the shot to complete
    sleep(0.5)


coords = (1500, 775)
for _ in range(1):
    shot_coords = (coords[0] + np.random.randint(-50, 50), coords[1] + np.random.randint(-50, 50))
    shoot_ball(shot_coords, power=50)
    sleep(3)
