import os
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from preprocessing import get_perspective_transformation_matrix

mutex = threading.Lock()
def del_line():
    print("\033[1A\x1b[2K", end="")
class ProgressWatcher:
    def __init__(self, folder_in: str, folder_out: str, num_threads: int) -> None:
        self.folder_in = folder_in
        self.folder_out = folder_out
        self.num_threads = num_threads
        
        self.source_files = os.listdir(folder_in)
        self.lengths = np.zeros(len(self.source_files)) # Init to 0
        self.progress = np.zeros(len(self.source_files)) # Init to 0
        self.matrix_acquired = False
        
        self.process_files_in_parallel()
        cv2.destroyAllWindows()
        
    def process(self, index: int) -> None:
        """Wrapper around the pipeline.

        Args:
            index (int): Index of the video to process.
        """
        path = os.path.join(self.folder_in, self.source_files[index])
        try:
            M = get_perspective_transformation_matrix(path)
            self.warp_video(index, M)
        except Exception as e:
            print(f"Error processing {path}: {e}")
        
    
    def warp_video(self, video_index: int, matrix: str):
        """Transform each frame of the video based on the transformation matrix and center crop to the arena shape.

        Args:
            video_index (int): Relative position of the video.
            matrix (ndarray): Perspective transformation matrix, obtained by the rest of the pipepile.

        Raises:
            RuntimeError: Video not found or corrupted.
        """
        path = os.path.join(self.folder_in, self.source_files[video_index])
        
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
        writer = cv2.VideoWriter(os.path.join(self.folder_out, f'{Path(path).stem}.mp4'), fourcc, fps, size_new)
        
        # Main video transform loop
        mutex.acquire()
        self.matrix_acquired = True
        self.progress[video_index] = 0
        self.lengths[video_index] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mutex.release()
        
        while cap.isOpened():
            # Read the old video
            ret, frame = cap.read()
            mutex.acquire()
            self.progress[video_index] += 1
            mutex.release()
            if ret == True:
                # Transform and write the new video
                frame = cv2.warpPerspective(frame, matrix, size_old)[y0-B:y1+B, x0-B:x1+B, :]
                writer.write(frame)
            else:
                break

        # Cleanup
        cap.release()
        writer.release()

    def writer(self):
        # Counts
        N_remaining = len(self.source_files)
        N_done = 0
        
        # Speed up for not checking finished files
        skip_array = [False] * N_remaining
        
        # Number of lines currently printed
        n_lines_in_message = 0
        
        # Clear screen
        print("\033[H\033[J", end="")
        
        while True:
            # Current files
            files_in_progress = []
            N_processing = 0
            
            mutex.acquire()
            for i, file in enumerate(self.source_files):
                if skip_array[i]: continue
                
                # All started or finished files
                if self.progress[i] > 0:
                    # All finished files
                    if self.progress[i] >= self.lengths[i] - 1:
                        N_done += 1
                        N_remaining -= 1
                        skip_array[i] = True
                    else: # Files in progress
                        N_processing += 1
                        files_in_progress.append((file, self.progress[i], self.lengths[i]))
            mutex.release()
            
            
            # Printing
            for i in range(n_lines_in_message): del_line() # Clear lines
            
            # Construct the print message
            out_str  = f"Remaining: \033[91m{N_remaining - N_processing}\033[0m | Processing: \033[94m{N_processing}\033[0m | Done: \033[92m{N_done}\033[0m\n"
            out_str += (len(out_str) - 27) * "-" + "\n"
            for file, progress, length in files_in_progress:
                file = f"\033[93m{file}\033[0m"
                out_str += f"{file}: {100 * (progress + 1) / length:.2f}%\n"
            print(out_str, end="")
            n_lines_in_message = 2 + len(files_in_progress)

            # Final message
            if N_remaining == 0:
                print("\033[92mAll done\033[0m\n")
                break
            
            # Reduce the call frequency of the writting thread
            time.sleep(1)


    def process_files_in_parallel(self) -> None:
        """Parallel wrapper around the processing pipeline.

        Args:
            file_list (list[str]): List of all videos to be processed.
            folder_out (str): Folder where the output videos are saved.
            num_threads (int): Number of thread to be used in parallel.
        """
        with ThreadPoolExecutor(self.num_threads) as executor:
            # Submitting each file to be processed in parallel
            futures = [executor.submit(self.writer)]
            futures.extend([executor.submit(self.process, i) for i in range(len(self.source_files))])

            # Wait for all tasks to complete
            for future in futures:
                future.result()
            
 
def main(parser: ArgumentParser) -> None:
    # Parser parsing
    args = parser.parse_args()
    folder_in = args.folder_in
    folder_out = args.folder_out
    num_threads = args.num_threads
    
    # Check if source folder exists and create output folder if needed
    if not os.path.exists(folder_in):
        raise RuntimeError()
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    
    ProgressWatcher(folder_in, folder_out, num_threads)

if __name__ == "__main__":
    parser = ArgumentParser(prog="NSD video preprocessing",
                            description="Loads and transforms all the videos in the given folder to have a square arena, with normalized size.")
    parser.add_argument("--folder_in", type=str, default="data")
    parser.add_argument("--folder_out", type=str, default="data_out")
    parser.add_argument("--num_threads", type=int, default=None)
    main(parser)
