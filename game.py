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

    cue_ball = find_cue_ball(img, circles)

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

def conv_coord_from_cropped_to_full(coord):
    ''' Convert coordinates from cropped image to full image '''
    # add 15 px to x and y coords
    coord = (coord[0] + constants['playable_area']['top_left'][0] + 15, coord[1] + constants['playable_area']['top_left'][1] + 15)
    return np.array(coord)

def find_cue_ball(img, circles):
    ''' Given image and circle coords, find circle with highest avg RGB (cue ball) '''
    cue_ball = None
    cue_ball_rgb = 0
    radius = constants['ball_radius']

    h, w = img.shape[:2]

    for circle in circles:
        x, y, _ = circle
        x, y = int(x), int(y)

        # Define bounding box limits and clamp to image bounds
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)

        crop = img[y1:y2, x1:x2]

        # New radius for the mask if crop size is smaller near edges
        crop_h, crop_w = crop.shape[:2]
        yy, xx = np.ogrid[:crop_h, :crop_w]
        center_y = crop_h // 2
        center_x = crop_w // 2
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2

        if crop.size == 0 or mask.shape != crop.shape[:2]:
            continue

        masked_pixels = crop[mask]
        avg_rgb = np.mean(masked_pixels, axis=0)
        brightness = np.sum(avg_rgb)

        if brightness > cue_ball_rgb:
            cue_ball_rgb = brightness
            cue_ball = (x, y)

    return cue_ball

def get_ghost_ball_coords(chosen_ball, pocket):
    """ Calculate the ghost ball coordinates for aiming. """
    chosen_ball, pocket = np.array(chosen_ball), np.array(pocket)
    pocket_to_ball = chosen_ball - pocket
    pocket_to_ball = pocket_to_ball / np.linalg.norm(pocket_to_ball) * 2 * constants['ball_radius']
    ghost_coords = chosen_ball + pocket_to_ball
    return np.array(ghost_coords, dtype=int)


def is_shot_possible(cue_ball, chosen_ball, ghost_coords, pocket):
    ''' 
    Given cue ball, chosen ball, and pocket, check if the shot is possible
    - check angle between cue ball, chosen ball, and pocket
    - check for interfering balls in both lines (cue to ball and ball to pocket)
    '''
    # calculate angle between cue ball, chosen ball, and pocket, ensure >= 100 deg
    cue_ball, chosen_ball, pocket = np.array(cue_ball), np.array(chosen_ball), np.array(pocket)
    ball_to_cue = cue_ball - ghost_coords
    ball_to_pocket = pocket - chosen_ball
    angle = np.arccos(np.dot(ball_to_cue, ball_to_pocket) / (np.linalg.norm(ball_to_cue) * np.linalg.norm(ball_to_pocket)))
    angle = np.degrees(angle)
    if angle < 100:
        return False

    # for middle pocket, check if the angle is too flat
    MIDDLE_POCKET_X = 1228
    MIN_STEEP_ANGLE = 30  # degrees off horizontal
    if pocket[0] == MIDDLE_POCKET_X:
        dx, dy = pocket - chosen_ball
        # compute angle between shot line and horizontal rail
        steepness = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if steepness < MIN_STEEP_ANGLE:
            # too flat against the rail → can’t pot
            return False
        
    # TODO: check for interfering balls in both lines (cue to ball and ball to pocket)

    return True

def pick_pocket(chosen_ball, cue_ball):
    ''' 
    Given chosen ball and cue ball, find the best pocket to aim for 
    ideas:
    - make sure angle is > 90 degrees (must be possible shot)
    - prioritize corner pockets (middle pockets must have approach angle within 45 deg of perp line)
    - check to see if there are interfering balls in both lines (cue to ball and ball to pocket)
    '''
    valid_pockets = []  # store (pocket_idx, ghost_coords, total ball travel distance)
    for pocket_idx in range(6):
        pocket = constants['pocket_aim_coords'][pocket_idx]
        ghost_coords = get_ghost_ball_coords(chosen_ball, pocket)
        possible = is_shot_possible(cue_ball, chosen_ball, ghost_coords, pocket)
        if possible:
            travel_distance = np.linalg.norm(chosen_ball - pocket) + np.linalg.norm(cue_ball - ghost_coords)
            valid_pockets.append((pocket_idx, ghost_coords, travel_distance))
    
    return min(valid_pockets, key=lambda x: x[2]) if valid_pockets else (None, None, None)
        

def manually_pick_ball_to_shoot():
    print("Spacebar to take screenshots and get list of balls. Press 'q' to quit.")

    pocket_aim_coords = constants['pocket_aim_coords']  # 6 (x, y) of places to aim

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

                cue_ball = find_cue_ball(img, circles)
                cue_ball = conv_coord_from_cropped_to_full(cue_ball)
                
                chosen_ball = show_image_and_click_to_choose(img, circles)
                if chosen_ball is None:
                    print("No ball chosen, skipping...")
                    continue
                chosen_ball = conv_coord_from_cropped_to_full(chosen_ball)
                sleep(0.5)

                _, aim_coords, _ = pick_pocket(chosen_ball, cue_ball)
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
