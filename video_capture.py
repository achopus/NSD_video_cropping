import os
import cv2
import numpy as np
import argparse

from scipy.ndimage import sobel
from sklearn.mixture import GaussianMixture
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

import time
from multiprocessing import Process


warnings.filterwarnings("ignore")

def get_lines(image):
    edges = np.sqrt(sobel(image, 0)**2 + sobel(image, 1)**2)
    eps = 0.05
    edges_log = np.log(edges + eps)
    edges_log = (edges_log - edges_log.min()) / (edges_log.max() - edges_log.min())
    rho = 0.5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 720  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 500  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments

    low_threshold = 200
    high_threshold = 255
    edges_canny = cv2.Canny(np.uint8(255 * edges_log), low_threshold, high_threshold)
    lines = cv2.HoughLinesP(edges_canny, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    return lines


def filter_lines(lines):
    if type(lines) != np.ndarray: return []
    lines_filtered = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if x1 < 300 or x2 > 1600: continue # Out-of-arena lines X
        #if 100 < y1 < 900 or 100 < y2 < 900: continue # In-arena lines
        if np.abs(x1 - x2) > 400 and np.abs(y1 - y2) > 60: continue # Incorrect angle - X
        if np.abs(y1 - y2) > 400 and np.abs(x1 - x2) > 60: continue # Incorrect ange - Y
        lines_filtered.append(line[0])
    return lines_filtered

def extract_arena(path, min_points=500):
    L = []
    cap = cv2.VideoCapture(path)
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    if not cap.isOpened(): raise RuntimeError
    frame_number = 15 * 60
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Time jump handling
    jump = 15 * 120 # 2 minute jump to the future
    n_jumps = 0
    brighteness_limit = 0.25
    bl_adjusted =  brighteness_limit
    
    frames_used = 0
    frames_needed = 15
    average_frame = np.zeros((frames_needed, H, W))
    while cap.isOpened():
        ret, img = cap.read()
        frame_number += 15
        print(f"\r{path} - {frame_number} - {len(L)}", end="")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if not ret: break
        gray = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)) / 255
        
        img_mean = np.mean(gray)
        if img_mean < bl_adjusted:
            frame_number += jump
            n_jumps += 1
            bl_adjusted = brighteness_limit * 0.95**n_jumps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            continue
        else:
            n_jumps = 0

        average_frame[frames_used, ...] = gray
        frames_used += 1
        if frames_used >= frames_needed:
            frames_used = 0
            average_frame = average_frame.mean(0)
            lines = get_lines(average_frame)
            lines = filter_lines(lines)
            """
            average_frame = np.repeat((255 * average_frame[:, :, None]).astype(np.uint8), 3, 2)
            for x0, y0, x1, y1 in lines: cv2.line(average_frame, (x0, y0), (x1, y1), color=(0, 0, 255))
            cv2.imshow("Frame", average_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            """
            average_frame = np.zeros((frames_needed, H, W)) 
            L.extend(lines)
        
        if len(L) >= min_points: break
    cap.release()
    return np.array(L), gray

def fit_arena(arena, n_iters = 10):
    arena_TL = arena[np.logical_and(arena[:, 0] < 750, arena[:, 1] < 600), :]
    arena_TR = arena[np.logical_and(arena[:, 0] > 750, arena[:, 1] < 600), :]
    arena_BL = arena[np.logical_and(arena[:, 0] < 750, arena[:, 1] > 600), :]
    arena_BR = arena[np.logical_and(arena[:, 0] > 750, arena[:, 1] > 600), :]
    arenas = [arena_TL, arena_TR, arena_BR, arena_BL]
    
    mm_sorted = np.zeros((4, 2))    
    for i, a in enumerate(arenas):
        for _ in range(n_iters):
            mixture = GaussianMixture(n_components=1, max_iter=1000, tol=1e-5, covariance_type='spherical').fit(a)
            density = mixture.score_samples(a)
            density_threshold = np.percentile(density, 4)
            a = a[density > density_threshold, :]
        mm = mixture.means_
        mm_sorted[i, :] = mm    
    return mm_sorted

def visualize_result(gray, arena, mm_sorted):
    plt.figure(figsize=(12, 12))
    plt.imshow(gray, cmap='gray')
    plt.scatter(arena[:, 0], arena[:, 1], marker='.', c='yellow', alpha=0.25)
    plt.plot(mm_sorted[:, 0], mm_sorted[:, 1], color='red')
    plt.plot(mm_sorted[[0, -1], 0], mm_sorted[[0, -1], 1], color='red')
    plt.scatter(mm_sorted[:, 0], mm_sorted[:, 1], c='orange', marker='o', s=50)
    plt.show()


def points_to_line(line):
    x0, y0, x1, y1 = line
    
    if x0 == x1:
        return None, x1  # The line is vertical, no slope, x = constant
    
    # Calculate the slope (m)
    m = (y1 - y0) / (x1 - x0)
    
    # Calculate the y-intercept (b)
    b = y0 - m * x0
    
    return m, b


def find_intersection(m1, b1, m2, b2):
    # Check if the lines are parallel
    if m1 == m2:
        return 0, 0, False
    
    if (m1 != None) and (m2 != None):
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    elif m1 != None and m2 == None:
        y = b2
        x = (b1 - b2) / m1
    elif m1 == None and m2 != None:
        y = b1
        x = (b2 - b1) / m2
    else:
        print(m1, b1, m2, b2)
    
    return x, y, True

def extract_points(lines):
    M_cos = np.zeros((len(lines), len(lines)))
    for i, li in enumerate(lines):
        for j, lj in enumerate(lines):
            Xi = li[0] - li[2]
            Yi = li[1] - li[3]
            Xj = lj[0] - lj[2]
            Yj = lj[1] - lj[3]
            c = (Xi * Xj + Yi * Yj) / (np.sqrt(Xi**2 + Yi**2) * np.sqrt(Xj**2 + Yj**2))
            M_cos[i, j] = c
    points = []
    for i, line_i in enumerate(lines):
        mi, bi = points_to_line(line_i)
        for j, line_j in enumerate(lines):
            if np.abs(M_cos[i, j]) > 0.01: continue
            mj, bj = points_to_line(line_j)
            xi, yi, flag = find_intersection(mi, bi, mj, bj)
            if xi < 0 or yi < 0: continue
            if xi > 1920 or yi > 1080: continue
            if xi > 1500 or xi < 300: continue
            if flag: points.append((xi, yi))
        
    return np.array(points)

def get_perspective_transform(folder_in: str, file: str, folder_out: str, min_points: int, n_outlier_iters: int):
    path = os.path.join(folder_in, file)
    lines, frame = extract_arena(path, min_points)
    points = extract_points(lines)
    """
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
    """
    corners = fit_arena(points, n_outlier_iters)
    DEFINED_CORNERS = np.array([[ 400,   50],
                                [1350,   50],
                                [1350, 1000],
                                [ 400, 1000]])

    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(DEFINED_CORNERS))
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], marker='o', c='red')
    plt.plot(corners[:, 0], corners[:, 1], color='red')
    plt.plot(corners[[0, -1], 0], corners[[0, -1], 1], color='red')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(folder_out, f"{folder_in}_{Path(file).stem}.png"), dpi=300, bbox_inches='tight')
    np.save(os.path.join(folder_out, f"{folder_in}_{Path(file).stem}.npy"), M)
    print(f"Done: {file}")
    return M, frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_in", default="videos", type=str)
    parser.add_argument("--folder_out", default="out", type=str)
    args = parser.parse_args()
    folder_in = args.folder_in
    folder_out = args.folder_out
    n_lines = 500
    n_outlier_iters = 5
    files = os.listdir(folder_in)
    files = [file for file in files if (not os.path.isdir(file) and '.mp4' in file)]
    N = len(files)
    
    for file in files:
        get_perspective_transform(folder_in, file, folder_out, n_lines, n_outlier_iters)