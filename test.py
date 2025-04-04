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

def setup():
    # Activate the window
    try:
        window = pgw.getWindowsWithTitle('8Ball')[0]
        window.activate()
    except:
        print("Window not found or couldn't be activated")
        # run scrcpy --no-audio --window-title=8Ball
        subprocess.Popen(['scrcpy', '--no-audio', '--fullscreen', '--window-title=8Ball'])
        sleep(4)  # Give scrcpy some time to open the window
        # wait until the window is available
        for _ in range(10):  # Try up to 10 times
            if pgw.getWindowsWithTitle('8Ball'):
                window = pgw.getWindowsWithTitle('8Ball')[0]
                window.activate()
                break
                sleep(1)
        else:
            print("Failed to find or activate the '8Ball' window after 10 attempts")

def collect_data():
    with mss.mss() as sct:
        target_monitor = sct.monitors[1]  # Primary monitor
        
        # create capture area using actual monitor dimensions
        monitor = {
            "top": target_monitor["top"], 
            "left": target_monitor["left"], 
            "width": target_monitor["width"], 
            "height": target_monitor["height"]
        }
        
        # every 3s for 10 times
        for i in range(10):
            sct_img = sct.grab(monitor)
            img_cv2 = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            
            # write to file, save in screenshots folder
            screenshot_path = screenshot_dir / f'screenshot_{i}.png'
            cv2.imwrite(str(screenshot_path), img_cv2)
            print(f"Screenshot {i} taken")
            sleep(3)

def shoot_ball(coords, power=100):
    ''' Given (x, y) coordinates and power, click to aim and shoot the ball'''
    # click to aim
    pag.click(coords[0], coords[1])
    sleep(0.3)  # Wait for the aim to settle
    # drag power stick down to shoot
    pag.mouseDown(POWER_STICK_TOP[0], POWER_STICK_TOP[1])
    # calculate the drag distance based on power
    drag_distance = int((POWER_STICK_BOTTOM[1] - POWER_STICK_TOP[1]) * (power / 100))
    # drag down to shoot
    pag.moveTo(POWER_STICK_BOTTOM[0], POWER_STICK_TOP[1] + drag_distance, duration=0.25)
    pag.mouseUp()
    # wait for the shot to complete
    sleep(0.5)

# TODO: figure out how to determine if it's my turn to shoot (maybe amt of green in time bar is changing?)
# or just check the outline on the profile pic
# or check for motion? if not motion in game area, maybe is my turn to shoot

coords = (1500, 775)
for _ in range(1):
    shot_coords = (coords[0] + np.random.randint(-50, 50), coords[1] + np.random.randint(-50, 50))
    shoot_ball(shot_coords, power=50)
    sleep(3)
