import os
import cv2
import numpy as np
from numpy import ndarray

from scipy.ndimage import sobel
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# TODO Let user define the variables in this function
def get_lines(image: ndarray) -> ndarray:
    """Find lines in the given image

    Args:
        image (ndarray): Image to be processed.

    Returns:
        ndarray: N x (x1, y1, x2, y2) lines array
    """
    
    
    # Calculate edges
    edges = np.sqrt(sobel(image, 0)**2 + sobel(image, 1)**2)
    eps = 0.05
    edges_log = np.log(edges + eps)
    edges_log = (edges_log - edges_log.min()) / (edges_log.max() - edges_log.min())
    
    # Refine edges
    low_threshold = 200
    high_threshold = 255
    edges_canny = cv2.Canny(np.uint8(255 * edges_log), low_threshold, high_threshold)
    
    # Get lines in the image
    rho = 0.5
    theta = np.pi / 720
    threshold = 30 
    min_line_length = 850
    max_line_gap = 150
    lines = cv2.HoughLinesP(edges_canny, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    return lines


def get_corners(lines: ndarray) -> list[tuple[float, float]]:
    """Get line endpoinds, after some filtering.

    Args:
        lines (ndarray): Array of the lines

    Returns:
        list[tuple[float, float]]: Filtered corners
    """
    if type(lines) != np.ndarray: return []
    C = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # TODO Generelize
            # Line filtering (sorry for magic varaibles - works for NSD setup, change if needed)
            if x1 < 300 or x2 > 1600: continue # Out-of-arena lines X
            if 100 < y1 < 900 or 100 < y2 < 900: continue # In-arena lines
            if np.abs(x1 - x2) > 400 and np.abs(y1 - y2) > 30: continue # Incorrect angle - X
            if np.abs(y1 - y2) > 400 and np.abs(x1 - x2) > 30: continue # Incorrect ange - Y

            # Save line end points
            C.append((x1, y1))
            C.append((x2, y2))
            
    return C

def extract_arena(path: str, min_points: int = 500) -> tuple[ndarray, ndarray]:
    """Extract the predicted corners from several frames of the video.

    Args:
        path (str): Path to video
        min_points (int, optional): Minimal number of corner points extracted. Defaults to 500.

    Raises:
        RuntimeError: Video not found or corrupted.

    Returns:
        tuple[ndarray, ndarray]: Array of corners and a sample frame
    """
    
    C = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError
    frame_number = 0
    
    # Time jump handling
    # TODO Generelize the time jump variables
    jump = 15 * 120 # 2 minute jump to the future
    n_jumps = 0
    brighteness_limit = 0.25
    bl_adjusted =  brighteness_limit

    # Play video
    while cap.isOpened():
        ret, img = cap.read()
        frame_number += 1
        if not ret: break
        gray = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)) / 255
        
        # Time jump - find higher brightness part of the video
        img_mean = np.mean(gray)
        if img_mean < bl_adjusted:
            frame_number += jump
            n_jumps += 1
            bl_adjusted = brighteness_limit * 0.95**n_jumps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            continue
        else:
            n_jumps = 0

        # Corners
        lines = get_lines(gray)
        corners = get_corners(lines)
        C.extend(corners)
        if len(C) >= min_points: break
    
    cap.release() # Cleanup
    
    return np.array(C), gray

def fit_arena(arena: ndarray, n_iters: int = 10) -> ndarray:
    """Find the actual centers from the array of all endpoints.

    Args:
        arena (ndarray): Array of all extracted corners which define the arena shape.
        n_iters (int, optional): Number of outlier removal iterations. Defaults to 10.

    Returns:
        ndarray: 4x2 Fitted corners of the arena, going clock-wise starting with top-left.
    """
    
    # Outlier removal and mixture fitting, each mixture is corresponds to a single corner
    for _ in range(n_iters):
        mixture = GaussianMixture(n_components=4, max_iter=1000, tol=1e-5, covariance_type='spherical').fit(arena)
        density = mixture.score_samples(arena)
        density_threshold = np.percentile(density, 4)
        arena = arena[density > density_threshold, :]
    mm = mixture.means_
    
    # Assign each found mixture to the correct corner (EM outputs are returned in a random order)
    # TODO Generelize the definitions the positions of the points
    mm_sorted = np.zeros((4, 2))    
    TL = np.where(np.logical_and(mm[:, 0] < 800, mm[:, 1] < 600))[0][0]
    TR = np.where(np.logical_and(mm[:, 0] > 800, mm[:, 1] < 600))[0][0]
    BL = np.where(np.logical_and(mm[:, 0] < 800, mm[:, 1] > 600))[0][0]
    BR = np.where(np.logical_and(mm[:, 0] > 800, mm[:, 1] > 600))[0][0]
    mm_sorted[0, :] = mm[TL, :]
    mm_sorted[1, :] = mm[TR, :]
    mm_sorted[2, :] = mm[BR, :]
    mm_sorted[3, :] = mm[BL, :]
    
    return mm_sorted

def visualize_result(image: ndarray, arena: ndarray, mm_sorted: ndarray, path: str) -> None:
    """Visualize the arena, all predicted corners and the refined solution.

    Args:
        image (ndarray): Background frame, which should corresponds to the given 
        arena (ndarray): All predicted corners
        mm_sorted (ndarray): Refined corners
        path (str): Filepath to the original video
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    plt.scatter(arena[:, 0], arena[:, 1], marker='.', c='yellow', alpha=0.25)
    plt.plot(mm_sorted[:, 0], mm_sorted[:, 1], color='red')
    plt.plot(mm_sorted[[0, -1], 0], mm_sorted[[0, -1], 1], color='red')
    plt.scatter(mm_sorted[:, 0], mm_sorted[:, 1], c='orange', marker='o', s=50)
    plt.savefig(f"{Path(path).stem}.png")

def pipeline(path: str, visualize: bool = False, min_points: int = 500) -> ndarray:
    """Find the four corners of the arena.
    Args:
        path (str): Filepath to the video.
        visualize (bool, optional): Save the example frame as a image. Defaults to False.
        min_points (int, optional): Minimal number of corner predictions to extract. Defaults to 500.

    Returns:
        ndarray: Corners of the arena.
    """
    arena, gray = extract_arena(path, min_points)
    corners = fit_arena(arena)
    if visualize: visualize_result(gray, arena, corners, path)
    return corners

def get_perspective_transformation_matrix(path: str) -> ndarray:
    """Calculate the perspective transform for the given video

    Args:
        path (str): Filepath to the video.

    Returns:
        ndarray: Perspective transformation matrix
    """
    # TODO Let user specify own corners
    DEFINED_CORNERS = np.array([[ 400,   50],
                                [1350,   50],
                                [1350, 1000],
                                [ 400, 1000]])
    corners = pipeline(path)
    return cv2.getPerspectiveTransform(np.float32(corners), np.float32(DEFINED_CORNERS))

def warp_video(path: str, folder_out: str, matrix: str):
    """Transform each frame of the video based on the transformation matrix and center crop to the arena shape.

    Args:
        path (str): Filepath to the video.
        folder_out (str): Folder, where the output videos are saved.
        matrix (str): Perspective transformation matrix, obtained by the rest of the pipepile.

    Raises:
        RuntimeError: Video not found or corrupted.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError
    
    # Arena bounding box
    y0 = 50
    y1 = 1000
    x0 = 400
    x1 = 1350
    B = 50 # boundary

    # Source video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    size_old = (int(cap.get(3)), int(cap.get(4)))

    # Transformed video info
    frame_width = (x1 - x0) + 2 * B
    frame_height = (y1 - y0) + 2 * B
    size_new = (frame_width, frame_height)
    
    # Video writing for the new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(folder_out, f'{Path(path).stem}.mp4'), fourcc, fps, size_new)
    
    # Main video transform loop
    while cap.isOpened():
        # Read the old video
        ret, frame = cap.read()
        if ret == True:
            # Transform and write the new video
            frame = cv2.warpPerspective(frame, matrix, size_old)[y0-B:y1+B, x0-B:x1+B, :]
            writer.write(frame)
        else:
            break

    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
            
