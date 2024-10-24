import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

points = []
go_next = False
progress = 0
def get_point(video_path, folder_out, folder_corners):
    global points, go_next, progress
    # Global variables to store points

    # Callback function to handle mouse events
    def select_points(event, x, y, flags, param):
        global points, go_next
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:  # Only allow 4 points
                points.append((x, y))
                print(f"Point selected: ({x}, {y})| {points}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = []
            print(f"Points reset: {points}")
        elif event == cv2.EVENT_MBUTTONDOWN:
            go_next = True
    
    
    def plot_text(frame, points):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        org = (15, 15)
        color = (0, 0, 255)
        thicknes = 1
        frame_copy = frame.copy()
        cv2.putText(frame_copy, progress, org, font, fontScale, color, thicknes, cv2.LINE_AA, False)
        for point in points:
            cv2.circle(frame_copy, point, radius=5, color=color, thickness=-1)
        return frame_copy

    # Open video file or capture from camera
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read the first frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        exit()

    # Create a window and set mouse callback function
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', select_points)

    # Display the frame and wait for 4 points to be selected
    frame_number = 0
    while True:
        # Display the frame
        cv2.imshow('Frame', plot_text(frame, points))
        
        # Check if 4 points have been selected
        if len(points) == 4:
            break
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Reset points
        if go_next:
            go_next = False
            frame_number += 15 * 30
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame_pos = cap.read()
            if ret: frame = frame_pos

    # Release the video capture and close windows
    cap.release()
    #cv2.destroyAllWindows()
    cv2.imshow('Frame', plot_text(frame, points))
    points_out = points
    print(f"{video_path} - {points}")
    go_next = False
    points = []
    corners = np.array(points_out)
    plt.figure()
    plt.imshow(frame.mean(-1), cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], marker='o', c='red')
    plt.plot(corners[:, 0], corners[:, 1], color='red')
    plt.plot(corners[[0, -1], 0], corners[[0, -1], 1], color='red')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{os.path.join(folder_out, Path(video_path).stem)}.png", dpi=300, bbox_inches='tight')
    DEFINED_CORNERS = np.array([[ 400,   50],
                                [1350,   50],
                                [1350, 1000],
                                [ 400, 1000]])
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(DEFINED_CORNERS))
    np.save(f"{os.path.join(folder_out, Path(video_path).stem)}.npy", M)
    np.save(f"{os.path.join(folder_corners, Path(video_path).stem)}.npy", corners)
    
if __name__ == "__main__":
    folder_in = "videos"
    folder_out = "out_test"
    folder_corners = "points"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    files = os.listdir(folder_in)
    for i, file in enumerate(files):
        progress = f"Progress: {100 * (i+1) / len(files):.2f}% | {file}"
        video_path = os.path.join(folder_in, file)
        get_point(video_path, folder_out, folder_corners)