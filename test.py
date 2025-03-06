import pygetwindow as pgw
import mss
import cv2
import numpy as np

# Activate the window
try:
    window = pgw.getWindowsWithTitle('8Ball')[0]
    window.activate()
    window.maximize()
except:
    print("Window not found or couldn't be activated")
    # run scrcpy --no-audio --window-title=8Ball

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
    
    # Capture the screen
    sct_img = sct.grab(monitor)
    
    # Convert to CV2 format
    img_cv2 = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
    
    # Write to file
    cv2.imwrite('screenshot.png', img_cv2)

