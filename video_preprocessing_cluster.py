import os
import time
from pathlib import Path
from argparse import ArgumentParser

import cv2
from preprocessing import get_perspective_transformation_matrix

from multiprocessing import Process, Array, Lock

class color:
    """Simple lookup table"""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def del_line():
    """Terminal delete line"""
    print("\033[1A\x1b[2K", end="")

    
def pipeline_wrapper(index: int, lengths_array, progress_array, runtime_array, folder_in, folder_out, source_files, lock) -> None:
    """Wrapper around the pipeline.

    Args:
        index (int): Index of the video to process.
    """
    path = os.path.join(folder_in, source_files[index])
    try:
        M = get_perspective_transformation_matrix(path)
        warp_video(index, source_files[index], M, lengths_array, progress_array, runtime_array, folder_in, folder_out, lock)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        
    
def warp_video(video_index: int, path: str, matrix: str, lengths_array, progress_array, runtime_array, folder_in, folder_out, lock):
    """Transform each frame of the video based on the transformation matrix and center crop to the arena shape.

    Args:
        video_index (int): Relative position of the video.
        matrix (ndarray): Perspective transformation matrix, obtained by the rest of the pipepile.

    Raises:
        RuntimeError: Video not found or corrupted.
    """
    path = os.path.join(folder_in, path)
    
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
    with lock:
        progress_array[video_index] = 0
        lengths_array[video_index] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    t0 = time.time()
    while cap.isOpened():
        # Read the old video
        ret, frame = cap.read()
        with lock:
            progress_array[video_index] += 1
            runtime_array[video_index] = time.time() - t0
        if ret == True:
            # Transform and write the new video
            frame = cv2.warpPerspective(frame, matrix, size_old)[y0-B:y1+B, x0-B:x1+B, :]
            writer.write(frame)
        else:
            break

    # Cleanup
    cap.release()
    writer.release()

def writer(lengths_array, progress_array, runtime_array, source_files, lock):
    # Counts
    N_remaining = len(source_files)
    N_done = 0
    
    # Speed up for not checking finished files
    skip_array = [False] * N_remaining
    
    # Number of lines currently printed
    n_lines_in_message = 0
    filename_length = max([len(f) for f in source_files])
    
    # Clear screen
    print("\033[H\033[J", end="")
    t0 = time.time()
    while True:
        # Current files
        files_in_progress = []
        N_processing = 0
        

        for i, file in enumerate(source_files):
            with lock:
                progress = progress_array[i]
                lengths = lengths_array[i]
                runtime = runtime_array[i]
            
            if skip_array[i]: continue
            
            # All started or finished files
            if progress > 0:
                # All finished files
                if progress >= lengths - 1:
                    N_done += 1
                    N_remaining -= 1
                    skip_array[i] = True
                else: # Files in progress
                    N_processing += 1
                    files_in_progress.append((file, progress, lengths, runtime))
        
        
        # Printing
        for i in range(n_lines_in_message): del_line() # Clear lines
        
        # Construct the print message
        out_str = f"{color.BOLD}Total runtime: {color.YELLOW}{time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}{color.END}\n"
        out_str += f"{color.BOLD}Remaining: \033[91m{N_remaining - N_processing}\033[0m │ {color.BOLD}Processing: \033[94m{N_processing}\033[0m │ {color.BOLD}Done: \033[92m{N_done}\033[0m\033[0;0m\n\n"
        out_str += "┌" + 32 * "─" + "┬" + 10 * "─" + "┬" + 18 * "─" + "┬" + 14 * "─" + "┬" + 16 * "─" + "┬" + 12 * "─" + "┐\n"
        out_str += f"│{color.BOLD}{color.YELLOW}{'Name'.center(filename_length + 1)}{color.END} │ {color.BOLD}{color.GREEN}Progress{color.END} │ {color.BOLD}{color.PURPLE}Frames processed{color.END} │ {color.BOLD}{color.CYAN}Elapsed time{color.END} │ {color.BOLD}{color.DARKCYAN}Remaining time{color.END} │ {color.BOLD}{color.RED}{'Speed'.center(10)}{color.END} │\n"
        out_str += "├" + 32 * "─" + "┼" + 10 * "─" + "┼" + 18 * "─" + "┼" + 14 * "─" + "┼" + 16 * "─" + "┼" + 12 * "─" + "┤\n"
        for i, (file, progress, length, runtime) in enumerate(files_in_progress):
            file: str = f"\033[93m{file.ljust(filename_length)}\033[0m"
            percent_done ='{:.2f}'.format(round(100 * (progress + 1) / length, 2)).rjust(7)
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(runtime)).center(12)
            remaining_time = time.strftime("%H:%M:%S", time.gmtime(runtime / (progress + 1) * (length - progress - 1))).center(14)
            its_p_s = '{:.2f} it/s'.format((progress + 1) / runtime).ljust(10)
            progress = str(int(progress + 1)).rjust(6)
            length = str(int(length)).rjust(6)
            out_str += f"│ {file} │ {color.GREEN}{percent_done}%{color.END} │ {color.PURPLE}{progress} / {length}{color.END}  │ {color.CYAN}{elapsed_time}{color.END} │ {color.DARKCYAN}{remaining_time}{color.END} │ {color.RED}{its_p_s}{color.END} │\n"
        out_str += "└" + 32 * "─" + "┴" + 10 * "─" + "┴" + 18 * "─" + "┴" + 14 * "─" + "┴" + 16 * "─" + "┴" + 12 * "─" + "┘\n"
        print(out_str, end="")
        n_lines_in_message = 7 + len(files_in_progress)

        # Final message
        if N_remaining == 0:
            for _ in range(n_lines_in_message - 3): del_line()
            print(f"{color.BOLD}{color.GREEN}All done{color.END}\n")
            break
        
        # Reduce the call frequency of the writting thread
        time.sleep(1)
            
 
def main(parser: ArgumentParser) -> None:
    # Parser parsing
    args = parser.parse_args()
    folder_in = args.folder_in
    folder_out = args.folder_out
    max_running_processes = args.num_workers
    
    # Check if source folder exists and create output folder if needed
    if not os.path.exists(folder_in):
        raise RuntimeError()
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    source_files = os.listdir(folder_in)
    N = len(source_files)
    max_running_processes = min(N, max_running_processes)
        
    lengths = Array('i', [0] * N)
    progress = Array('i', [0] * N)
    runtime = Array('f', [0] * N)
    lock = Lock()
    

    process_queue: list[Process] = []  # Queue to hold running processes
    current_process = 0  # Keep track of the current process number

    # Start writer        
    writer_process = Process(target=writer, args=(lengths, progress, runtime, source_files, lock))
    writer_process.start()
    process_queue.append(writer_process)
    current_process += 1
    
    file_id = 0
    # Initialize first batch of processes
    while current_process <= max_running_processes:
        process = Process(target=pipeline_wrapper, args=(file_id, lengths, progress, runtime, folder_in, folder_out, source_files, lock))
        process.start()
        process_queue.append(process)
        current_process += 1
        file_id += 1
        

    # Dynamically add new processes as old ones finish
    
    while True:
        # Check for any completed processes
        for process in process_queue:
            if not process.is_alive():
                process_queue.remove(process)  # Remove completed process
                # Start a new process to replace the completed one
                if file_id == N: break
                process = Process(target=pipeline_wrapper, args=(file_id, lengths, progress, runtime, folder_in, folder_out, source_files, lock))
                process.start()
                process_queue.append(process)
                current_process += 1
                file_id += 1
                
                break  # Avoid modifying process_queue while iterating

        # Avoid busy-waiting by adding a short sleep
        if file_id == N: break
        time.sleep(0.5)

    # Wait for all remaining processes to complete
    for process in process_queue:
        process.join()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser(prog="NSD video preprocessing",
                            description="Loads and transforms all the videos in the given folder to have a square arena, with normalized size.")
    parser.add_argument("--folder_in", type=str, default="data")
    parser.add_argument("--folder_out", type=str, default="data_out")
    parser.add_argument("--num_workers", type=int, default=1)
    main(parser)
