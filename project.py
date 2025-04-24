import cv2
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
from models import BallCNN

root = Path(__file__).parent
constants = json.load(open(root / 'constants.json'))

def compute_ratios(img, use_blue=False):
    """
    Generate pixel-wise red-green and blue-green ratios.
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    if use_blue:
        B[B == 0] = 1e-6
        return R/B, G/B
    G[G == 0] = 1  # avoid division by zero
    return R/G, B/G

def compute_background_thresholds(ratios, k):
    """
    Compute the ratio thresholds within k standard deviations of the mean background value.
    Here we assume green is the most abundant color in the background.
    param k: the maximal admissible number of standard deviations.
    """
    flat = ratios.flatten()
    flat = flat[flat < 5] # filter outliers for fit
    mu, std = norm.fit(flat)
    return mu - k * std, mu + k * std, mu, std

def create_foreground_mask(img, ratio_1, ratio_2, thresh_1, thresh_2):
    """
    Create a foreground mask including all pixels not within an acceptable range of background
    r/g and b/g values.
    param ratio_1, ratio_2: ratio arrays (default red-green and blue-green; if use_blue is true,
                            red-blue and green-blue)
    param thresh_1, thresh_2: threshold intervals for background values.
    """
    mask_1 = (ratio_1 >= thresh_1[0]) & (ratio_1 <= thresh_1[1])
    mask_2 = (ratio_2 >= thresh_2[0]) & (ratio_2 <= thresh_2[1])
    background_mask = mask_1 & mask_2
    # keep black pixels (avg < 10), this is to avoid deleting the 8 ball
    background_mask = background_mask & (img.mean(axis=2) > 10)
    return (~background_mask * 255).astype(np.uint8)

def find_circles(mask, min_dist, canny, accum, min_radius, max_radius):
    """
    Find circles using Hough circles.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # Hough Circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,   
        param1=canny,        # Canny edge detection
        param2=accum,        # accumulator threshold
        minRadius=min_radius,
        maxRadius=max_radius
    )

    return circles

def generate_data(img, use_blue=False, k_1 = 1.2, k_2 = 1.5, min_dist=40, canny=140, accum=40, min_radius=20, max_radius=200):
    """
    Given an image, create bounding boxes for detected balls and create data images.
    """
    h, w = img.shape[:2]
    
    # Compute thresholds
    rg, bg = compute_ratios(img, use_blue=use_blue)
    rg_thresh = compute_background_thresholds(rg, k=k_1)
    bg_thresh = compute_background_thresholds(bg, k=k_2)

    # Create mask and find circles
    mask = create_foreground_mask(img, rg, bg, rg_thresh, bg_thresh)

    circles = find_circles(mask, min_dist, canny, accum, min_radius, max_radius)
    new_circles = []

    data = []
    circles = np.around(circles).astype(np.uint16)
    if circles is not None:
        for (x, y, _) in circles[0, :]:
            r = 28
            if r <= x <= w-1-r and r <= y <= h-1-r:
                cropped = img[y-r:y+r, x-r:x+r].copy()
                data.append(cropped)
                new_circles.append((x, y, r))
    
    return [new_circles], data

def preprocess_image(img, size=48, padding=0, thresh_1=(0.9,1.25), thresh_2=(0.65,1), close_size=3, open_size=3):
    """
    Preprocess provided image into black-and-white data ready for neural network
    """

    img = cv2.resize(img, (size-2*padding, size-2*padding))
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    ratio_1, ratio_2 = compute_ratios(img)
    mask_ball = (255*((ratio_1 >= thresh_1[0]) & (ratio_1 <= thresh_1[1]) & (ratio_2 >= thresh_2[0]) & (ratio_2 <= thresh_2[1]))).astype(np.uint8)
    kernel_close = np.ones((close_size, close_size),np.uint8)
    kernel_open = np.ones((open_size, open_size),np.uint8)

    # Apply morphology to clean up
    mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, kernel_close)
    mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_OPEN, kernel_open)

    # Crop circle shape
    y, x = np.ogrid[:size, :size]
    r = size/2-padding
    cx, cy = (size-1)/2, (size-1)/2
    circle_mask = (x - cx)**2 + (y - cy)**2 <= r**2

    return (circle_mask * mask_ball).astype(np.uint8)

def preprocess_file(impath, size=48, padding=0, thresh_1=(0.9,1.25), thresh_2=(0.65,1), close_size=3, open_size=3):
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img, size, padding, thresh_1, thresh_2, close_size, open_size).astype(np.float32)/255

def preprocess_data(data, size=48, padding=0, thresh_1=(0.9,1.25), thresh_2=(0.65,1), close_size=3, open_size=3):
    return [preprocess_image(img, size, padding, thresh_1, thresh_2, close_size, open_size) for img in data]


def find_cue_ball(img, circles):
    ''' Given image and circle coords, find circle with highest avg RGB (cue ball) '''
    cue_ball = None
    cue_ball_rgb = 0
    radius = constants['ball_radius']

    h, w = img.shape[:2]

    for circle in circles:
        x, y, _ = circle
        x, y = int(x), int(y)

        # clamp to image bounds
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)

        crop = img[y1:y2, x1:x2]

        # make sure don't crop near edges
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

def find_8_ball(img, circles):
    ''' Given image and circle coords, find circle with most pixels rgb < (25, 25, 25) '''
    eight_ball = None
    max_dark_pixels = 0
    radius = constants['ball_radius']
    
    h, w = img.shape[:2]

    for circle in circles:
        x, y, _ = circle
        x, y = int(x), int(y)

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)

        crop = img[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]
        yy, xx = np.ogrid[:crop_h, :crop_w]
        center_y = crop_h // 2
        center_x = crop_w // 2
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2

        if crop.size == 0 or mask.shape != crop.shape[:2]:
            continue

        masked_pixels = crop[mask]

        dark_pixels = np.sum(np.all(masked_pixels < 25, axis=1))

        if dark_pixels > max_dark_pixels:
            max_dark_pixels = dark_pixels
            eight_ball = (x, y)

    return eight_ball


def conv_coord_from_cropped_to_full(coord):
    ''' Convert coordinates from cropped image to full image '''
    # add 15 px to x and y coords
    coord = (coord[0] + constants['playable_area']['top_left'][0] + 15, coord[1] + constants['playable_area']['top_left'][1] + 15)
    return np.array(coord)



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

def label_balls(img, circles, data):
    ''' 
    Given img, circle coords, and cropped ball data, return dict with
    {
        'cue_ball': (x, y),
        '8_ball': (x, y),
        'aim_circle': (x, y),
        'stripes': [(x, y), (x, y), ...],
        'solids': [(x, y), (x, y), ...]
    }
    '''
    model = BallCNN()
    model.load_state_dict(torch.load('ball_type.pth'))

    cue_ball = find_cue_ball(img, circles)
    eight_ball = find_8_ball(img, circles)
    # cue_ball = conv_coord_from_cropped_to_full(cue_ball)
    # eight_ball = conv_coord_from_cropped_to_full(eight_ball)

    res = {
        'cue_ball': cue_ball,
        '8_ball': eight_ball,
        'aim_circle': None,
        'stripes': [],
        'solids': []
    }

    for i in range(len(circles)):
        if cue_ball is not None and np.array_equal(circles[i][:2], cue_ball):
            continue
        if eight_ball is not None and np.array_equal(circles[i][:2], eight_ball):
            continue

        ball = preprocess_image(data[i], size=56, padding=0,
                                     thresh_1=(0.5, 1.1), thresh_2=(0.9, 1.1),
                                     close_size=2, open_size=2)
        model.eval()
        with torch.no_grad():
            ball = torch.from_numpy(ball).unsqueeze(0).unsqueeze(0) / 255  # normalize like toTensor
            ball = ball.to('cpu')
            pred = model(ball)
            pred = torch.argmax(pred, dim=1).item()
            if pred == 0:
                res['stripes'].append(circles[i][:2])
            elif pred == 1:
                res['solids'].append(circles[i][:2])
            elif pred == 2:
                res['aim_circle'] = tuple(circles[i][:2])
        
    return res
    
        




