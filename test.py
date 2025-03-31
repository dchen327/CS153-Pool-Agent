import pygetwindow as pgw
import mss
import cv2
import numpy as np
import subprocess
from time import sleep
from pathlib import Path

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
with mss.mss() as sct:
    print("Available monitors: ", sct.monitors)
    target_monitor = sct.monitors[1]  # Primary monitor
    
    # Create the capture area using actual monitor dimensions
    monitor = {
        "top": target_monitor["top"], 
        "left": target_monitor["left"], 
        "width": target_monitor["width"], 
        "height": target_monitor["height"]
    }
    
    # every 3s for 10 times
    for i in range(10):
        # Capture the screen
        sct_img = sct.grab(monitor)
        
        # Convert to CV2 format
        img_cv2 = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        
        # Write to file, save in screenshots folder
        screenshot_path = screenshot_dir / f'screenshot_{i}.png'
        cv2.imwrite(str(screenshot_path), img_cv2)
        print(f"Screenshot {i} taken")
        sleep(3)

