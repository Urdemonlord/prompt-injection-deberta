"""
Script untuk mengunduh dataset lengkap (train.csv, val.csv, test.csv) dari GitHub Releases
"""

import os
import sys
import urllib.request
import zipfile

RELEASE_URL = "https://github.com/Urdemonlord/prompt-injection-deberta/releases/download/v1.0.0/dataset.zip"
TARGET_ZIP = "dataset.zip"

def reporthook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    downloaded_mb = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rUnduh: {downloaded_mb:.1f} MB / {total_mb:.1f} MB [{percent}%]")
    sys.stdout.flush()

def main():
    if os.path.exists("train.csv"):
        print("[OK] 'train.csv' sudah ada di lokal.")
        return

    print(f"Mengunduh dataset dari GitHub Releases:")
    print(f"URL: {RELEASE_URL}")
    
    if not os.path.exists(TARGET_ZIP):
        try:
            urllib.request.urlretrieve(RELEASE_URL, TARGET_ZIP, reporthook)
            print("\nUnduhan selesai!")
        except Exception as e:
            print(f"\n[ERROR] Gagal mengunduh: {e}")
            sys.exit(1)
    else:
        print(f"File zip '{TARGET_ZIP}' sudah ada di lokal.")

    print(f"Mengekstrak file CSV...")
    with zipfile.ZipFile(TARGET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print(f"[SUKSES] Dataset lengkap (train.csv, val.csv, test.csv) siap digunakan!")

if __name__ == "__main__":
    main()
