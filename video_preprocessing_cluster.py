import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from preprocessing import get_perspective_transformation_matrix, warp_video

def process(path: str, folder_out: str) -> None:
    """Wrapper function around the transformation pipeline. Needed for parallel processing.

    Args:
        path (str): Filepath to the video
        folder_out (str): Folder where the output videos are saved.
    """
    try:
        M = get_perspective_transformation_matrix(path)
        warp_video(path, folder_out, M)
    except Exception as e:
        print(f"Error processing {path}: {e}")


def process_files_in_parallel(file_list: list[str], folder_out: str, num_threads: int) -> None:
    """Parallel wrapper around the processing pipeline.

    Args:
        file_list (list[str]): List of all videos to be processed.
        folder_out (str): Folder where the output videos are saved.
        num_threads (int): Number of thread to be used in parallel.
    """
    with ThreadPoolExecutor(num_threads) as executor:
        # Submitting each file to be processed in parallel
        futures = [executor.submit(process, file, folder_out) for file in file_list]

        # Wait for all tasks to complete
        for future in futures:
            future.result()
            
def main(args: ArgumentParser) -> None:
    # Parser parsing
    folder_in = args.folder_in
    folder_out = args.folder_out
    num_threads = args.num_threads
    
    # Check if source folder exists and create output folder if needed
    if not os.path.exists(folder_in):
        raise RuntimeError()
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    
    # Get all videos to be transformed
    files = [os.path.join(folder_in, file) for file in os.listdir(folder_in)]
    
    # Transform files in parallel
    process_files_in_parallel(files, folder_out, num_threads)

if __name__ == "__main__":
    parser = ArgumentParser(prog="NSD video preprocessing",
                            description="Loads and transforms all the videos in the given folder to have a square arena, with normalized size.")
    parser.add_argument("--folder_in", type=str, default="data")
    parser.add_argument("--folder_out", type=str, default="data_out")
    parser.add_argument("--num_threads", type=int, default=None)
    main(parser)