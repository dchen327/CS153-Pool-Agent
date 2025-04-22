import pygetwindow as pgw
import mss
import cv2
import numpy as np
import subprocess
import pyautogui as pag
import keyboard
import random
import json
import project
import matplotlib.pyplot as plt
from time import sleep
from pathlib import Path


POWER_STICK_TOP = (112, 615)
POWER_STICK_BOTTOM = (112, 1250)

root = Path(__file__).parent
# Create screenshots directory if it doesn't exist
screenshot_dir = root / 'screenshots'
screenshot_dir.mkdir(exist_ok=True)
constants = json.load(open(root / 'constants.json'))

def setup():
    # activate the window
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
            quit()

def get_highest_screenshot_num():
    ''' Get the highest screenshot number in the screenshots directory '''
    # screenshot_count = highest number in folder + 1
    screenshot_count = 0
    for file in screenshot_dir.glob('screenshot_*.png'):
        try:
            num = int(file.stem.split('_')[1])
            if num >= screenshot_count:
                screenshot_count = num + 1
        except ValueError:
            pass
    return screenshot_count

def collect_data():
    print("Spacebar to take screenshots. Press 'q' to quit.")
    screenshot_count = get_highest_screenshot_num()
    print(f"Starting from screenshot number: {screenshot_count}")
    with mss.mss() as sct:
        # set monitor to be the one with the 8Ball window
        monitor = sct.monitors[1]  # Primary monitor

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
    pag.moveTo(coords[0], coords[1])
    pag.click()
    sleep(0.5)  # Wait for the aim to settle
    # drag power stick down to shoot
    pag.moveTo(POWER_STICK_TOP[0], POWER_STICK_TOP[1])
    pag.mouseDown()
    sleep(0.2)
    # calculate the drag distance based on power
    drag_distance = int((POWER_STICK_BOTTOM[1] - POWER_STICK_TOP[1]) * (power / 100))
    # drag down to shoot
    pag.moveTo(POWER_STICK_BOTTOM[0], POWER_STICK_TOP[1] + drag_distance, duration=0.5)
    sleep(0.2)
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

def take_screenshots_from_break():
    ''' Repeatedly create new games, break, and take a screenshot when balls stop moving '''
    screenshot_count = get_highest_screenshot_num()
    print(f"Starting from screenshot number: {screenshot_count}")
    for _ in range(10):
        # start new game
        # if there are any ads open, close them by searching for the x button
        try:
            x_button = pag.locateOnScreen(str(root / 'other_assets/x_button.png'), confidence=0.8)
            if x_button:
                center = pag.center(x_button)
                pag.moveTo(center)
                pag.click()
                sleep(1.5)
                print("Closed ad")
        except pag.ImageNotFoundException:
            pass
        

        pag.moveTo((2000, 1000))  # drag left to offline game
        pag.mouseDown()
        pag.moveTo((200, 1000), duration=0.5)
        pag.mouseUp()
        sleep(0.75)

        pag.moveTo(2135, 842)
        pag.click()
        sleep(0.75)
        pag.moveTo(960, 1200)
        pag.click()
        sleep(3)

        shoot_ball((1629, 888), power=random.randint(85, 100))
        sleep(10)

        # take screenshot
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img_cv2 = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            # write to file, save in screenshots folder
            screenshot_path = screenshot_dir / f'screenshot_{screenshot_count}.png'
            cv2.imwrite(str(screenshot_path), img_cv2)
            print(f"Screenshot {screenshot_count} taken")
            screenshot_count += 1

        # quit game
        pag.moveTo(110, 282)  # Move mouse to the position
        pag.click()
        sleep(1.75)
        pag.moveTo(200, 702)  # Move mouse to the position
        pag.click()
        sleep(1.75)
        pag.moveTo(1000, 1050)  # Move mouse to the position
        pag.click()
        sleep(3)

def show_image_and_click_to_choose(img, circles):
    fig, ax = plt.subplots(figsize=(12, 10))

    # Maximize the window
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')  # For TkAgg on Windows
    except AttributeError:
        try:
            manager.full_screen_toggle()  # For QtAgg or Mac/Linux
        except Exception as e:
            print("Fullscreen not supported:", e)

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cue_ball = project.find_cue_ball(img, circles)

    for idx, (x, y, r) in enumerate(circles):
        if (x, y) == cue_ball:
            ax.add_patch(plt.Circle((x, y), r, color='red', fill=False, linewidth=2))
        else:
            ax.add_patch(plt.Circle((x, y), r, color='lime', fill=False, linewidth=2))
            ax.text(x - 15, y + 15, str(idx + 1), color='white', fontsize=12, weight='bold')

    plt.title("Click a ball to choose")
    plt.axis('off')

    chosen_ball = None

    def onclick(event):
        nonlocal chosen_ball
        if event.xdata is None or event.ydata is None:
            return
        click_point = np.array([event.xdata, event.ydata])
        distances = [np.linalg.norm(click_point - np.array([x, y])) for (x, y, r) in circles]
        closest_idx = np.argmin(distances)
        chosen_ball = circles[closest_idx]
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()  # Blocks until the window is closed

    return chosen_ball

        

def manually_pick_ball_to_shoot():
    print("Spacebar to take screenshots and get list of balls. Press 'q' to quit.")

    with mss.mss() as sct:
        # set monitor to be the one with the 8Ball window
        monitor = sct.monitors[1]  # Primary monitor

        while True:
            key = keyboard.read_key()
            if key == 'space':
                sct_img = sct.grab(monitor)
                img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                img = img[constants['playable_area']['top_left'][1]:constants['playable_area']['bottom_right'][1],
                        constants['playable_area']['top_left'][0]:constants['playable_area']['bottom_right'][0]]
                # crop image by another 5px on each side
                img = img[15:-15, 15:-15]
                # show image
                circles, data = project.generate_data(img, use_blue=False, k_1=2.5, k_2=1.5,
                                            min_dist=20, canny=100, accum=18, min_radius=23, max_radius=27)
                circles = circles[0]

                cue_ball = project.find_cue_ball(img, circles)
                cue_ball = project.conv_coord_from_cropped_to_full(cue_ball)
                
                chosen_ball = show_image_and_click_to_choose(img, circles)
                if chosen_ball is None:
                    print("No ball chosen, skipping...")
                    continue
                chosen_ball = project.conv_coord_from_cropped_to_full(chosen_ball)
                sleep(0.5)

                _, aim_coords, _ = project.pick_pocket(chosen_ball, cue_ball)
                if aim_coords is None:
                    print("No valid pocket found, skipping...")
                    continue

                shoot_ball((aim_coords[0], aim_coords[1]), power=random.randint(60, 80))


            elif key == 'q':
                print('Quitting...')
                break


if __name__ == "__main__":
    setup() 
    # collect_data()
    # take_screenshots_from_break()
    manually_pick_ball_to_shoot()
