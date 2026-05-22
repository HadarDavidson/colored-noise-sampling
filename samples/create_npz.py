import os
import argparse
import datetime
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import concurrent.futures

def load_image(img_path):
    """
    Worker function to load a single image.
    Returns None if the file is missing or corrupted.
    """
    if not os.path.exists(img_path):
        return None
    
    try:
        sample_pil = Image.open(img_path)
        # Convert to uint8 numpy array
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        return sample_np
    except Exception as e:
        # Catch corrupted/partially written files if the run crashed mid-write
        print(f"\nWarning: Skipping corrupted image {img_path} - {e}")
        return None

def create_npz_from_sample_folder_mp(sample_dir, num=50000, num_workers=None):
    """
    Builds a single .npz file from a folder of .png samples using multiple CPUs,
    ignoring missing or corrupted files.
    """
    if not os.path.exists(sample_dir):
        raise ValueError(f"The directory '{sample_dir}' does not exist.")

    # 1. Pre-compute the exact list of file paths we need to load
    img_paths = [os.path.join(sample_dir, f"{str(i).zfill(6)}.png") for i in range(num)]
    
    # 2. Determine the optimal number of CPU workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1
        
    print(f"Scanning for {num} images using {num_workers} CPU workers...")

    # 3. Execute the image loading in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(load_image, img_paths, chunksize=250), 
                total=num, 
                desc="Decoding images"
            )
        )
        
    # --- NEW: Filter out missing/corrupted images ---
    valid_results = [res for res in results if res is not None]
    actual_num = len(valid_results)
    
    if actual_num == 0:
        raise ValueError("No valid images were found in the directory!")
        
    print(f"\nSuccessfully loaded {actual_num} valid images (Missing/Corrupted: {num - actual_num}).")
    print("Stacking arrays into contiguous memory block (this may take a moment)...")
    
    samples = np.stack(valid_results)
    
    # Free the temporary lists to save RAM before saving
    del results 
    del valid_results
    
    # Update the assert to check against actual_num instead of the requested num
    assert samples.shape == (actual_num, samples.shape[1], samples.shape[2], 3), f"Unexpected shape: {samples.shape}"
    
    # 4. Save the final .npz
    timestamp = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    clean_dir_path = os.path.normpath(sample_dir)
    npz_path = f"{clean_dir_path}.npz"
    
    print(f"Saving uncompressed arrays to disk...")
    np.savez(npz_path, arr_0=samples)
    print(f"Successfully saved .npz file to {npz_path} [shape={samples.shape}].")
    
    return npz_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiprocessing: Convert a folder of JiT-generated PNGs to an NPZ file.")
    parser.add_argument("sample_dir", type=str, help="Path to the directory containing the .png files")
    parser.add_argument("--num", type=int, default=50000, help="Maximum number of images to look for (default: 50000)")
    parser.add_argument("--workers", type=int, default=None, help="Specific number of CPU cores to use (default: All available)")
    
    args = parser.parse_args()
    
    create_npz_from_sample_folder_mp(args.sample_dir, num=args.num, num_workers=args.workers)