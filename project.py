import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_ratios(img, use_blue=False):
    """
    Generate pixel-wise red-green and blue-green ratios.
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    if use_blue:
        B[B == 0] = 1
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

def create_foreground_mask(ratio_1, ratio_2, thresh_1, thresh_2):
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
    mask = create_foreground_mask(rg, bg, rg_thresh, bg_thresh)
    circles = find_circles(mask, min_dist, canny, accum, min_radius, max_radius)

    data = []
    circles = np.around(circles).astype(np.uint16)
    if circles is not None:
        for (x, y, r) in circles[0, :]:
            if r <= x <= w-1-r and r <= y <= h-1-r:
                cropped = img[y-r:y+r, x-r:x+r].copy()
                data.append(cropped)
    return data

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

    return (~circle_mask * 128 + circle_mask * mask_ball).astype(np.uint8)

def preprocess_file(impath, size=48, padding=0, thresh_1=(0.9,1.25), thresh_2=(0.65,1), close_size=3, open_size=3):
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img, size, padding, thresh_1, thresh_2, close_size, open_size).astype(np.float32)/255

def preprocess_data(data, size=48, padding=0, thresh_1=(0.9,1.25), thresh_2=(0.65,1), close_size=3, open_size=3):
    return [preprocess_image(img, size, padding, thresh_1, thresh_2, close_size, open_size) for img in data]