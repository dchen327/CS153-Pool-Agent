import pygetwindow as pgw
import mss
import cv2
import numpy as np
import subprocess
from time import sleep

# Activate the window
try:
    window = pgw.getWindowsWithTitle('8Ball')[0]
    window.activate()
except:
    print("Window not found or couldn't be activated")
    # run scrcpy --no-audio --window-title=8Ball
    subprocess.run(['scrcpy', '--no-audio', '--fullscreen', '--window-title=8Ball'], check=True)
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
        
        # Write to file
        cv2.imwrite(f'screenshot_{i}.png', img_cv2)
        sleep(3)

