"""
Download Adult Dataset from UCI Repository

This script downloads the Adult dataset files directly from UCI repository.
"""

import urllib.request
import os


def download_adult_dataset():
    """Download Adult dataset files from UCI repository"""
    
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    files = {
        "adult.data": "adult.data",
        "adult.test": "adult.test",
        "adult.names": "adult.names"
    }
    
    print("Downloading Adult Dataset files...")
    
    for filename, save_as in files.items():
        url = base_url + filename
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, save_as)
            print(f"  ✓ Saved as {save_as}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
    
    print("\nDownload completed!")
    print("\nFiles downloaded:")
    for file in files.values():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  - {file} ({size:,} bytes)")


if __name__ == "__main__":
    download_adult_dataset()
