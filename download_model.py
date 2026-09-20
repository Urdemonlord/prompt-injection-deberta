"""
Script untuk mengunduh bobot model DeBERTa-v3 (best_model_biner) dari GitHub Releases
dan mengekstraknya ke direktori model.
"""

import os
import sys
import urllib.request
import zipfile

RELEASE_URL = "https://github.com/Urdemonlord/prompt-injection-deberta/releases/download/v1.0.0/best_model_biner.zip"
TARGET_ZIP = "best_model_biner.zip"
TARGET_DIR = "best_model_biner"

def reporthook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    downloaded_mb = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rUnduh: {downloaded_mb:.1f} MB / {total_mb:.1f} MB [{percent}%]")
    sys.stdout.flush()

def main():
    safetensor_path = os.path.join(TARGET_DIR, "model.safetensors")
    if os.path.exists(safetensor_path):
        print(f"[OK] Model weights sudah ada di '{safetensor_path}'.")
        return

    print(f"Mengunduh model weights dari GitHub Releases:")
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

    print(f"Mengekstrak ke folder '{TARGET_DIR}'...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    with zipfile.ZipFile(TARGET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    
    print(f"[SUKSES] Model siap digunakan di '{TARGET_DIR}/'!")

if __name__ == "__main__":
    main()
