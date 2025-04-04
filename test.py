import pygetwindow as pgw
import mss
import cv2
import numpy as np
import subprocess
import pyautogui as pag
import keyboard
from time import sleep
from pathlib import Path


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
    print("Spacebar to take screenshots. Press 'q' to quit.")
    # screenshot_count = highest number in folder + 1
    screenshot_count = 0
    for file in screenshot_dir.glob('screenshot_*.png'):
        try:
            num = int(file.stem.split('_')[1])
            if num >= screenshot_count:
                screenshot_count = num + 1
        except ValueError:
            pass
    print(f"Starting from screenshot number: {screenshot_count}")
    with mss.mss() as sct:
        target_monitor = sct.monitors[1]  # Primary monitor
        
        # create capture area using actual monitor dimensions
        monitor = {
            "top": target_monitor["top"], 
            "left": target_monitor["left"], 
            "width": target_monitor["width"], 
            "height": target_monitor["height"]
        }

        while True:
            key = keyboard.read_key()  # this blocks
            
            if key == 'space':
                sct_img = sct.grab(monitor)
                img_cv2 = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                
                # write to file, save in screenshots folder
                screenshot_path = screenshot_dir / f'screenshot_{screenshot_count}.png'
                cv2.imwrite(str(screenshot_path), img_cv2)
                print(f"Screenshot {screenshot_count} taken")
                screenshot_count += 1
            elif key == 'q':
                print('Quitting...')
                break

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

def test_shoot():
    coords = (1500, 775)
    for _ in range(1):
        shot_coords = (coords[0] + np.random.randint(-50, 50), coords[1] + np.random.randint(-50, 50))
        shoot_ball(shot_coords, power=50)
        sleep(3)
    

if __name__ == "__main__":
    setup()
    collect_data()
