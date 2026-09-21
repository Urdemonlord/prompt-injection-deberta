"""
Unduh model terjemahan lokal (MarianMT) untuk lapisan adaptasi multibahasa.

Dipakai menggantikan Google Translate yang selalu kena HTTP 429 dari IP VPS ini.
Model default: Helsinki-NLP/opus-mt-mul-en (banyak bahasa -> Inggris, ~310 MB).

Jalankan:  python3 download_mt_model.py
Hasil   :  ./mt_model/  (di-copy ke image oleh webapp/Dockerfile.deploy)
"""
import os
import sys

MODEL_ID = os.getenv("MT_MODEL_ID", "Helsinki-NLP/opus-mt-mul-en")
TARGET_DIR = os.getenv("MT_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mt_model"))

ALLOW = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "source.spm",
    "target.spm",
    "pytorch_model.bin",
]


def main() -> None:
    if os.path.exists(os.path.join(TARGET_DIR, "pytorch_model.bin")):
        print(f"[OK] Model MT sudah ada di '{TARGET_DIR}'.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub tidak terpasang. Jalankan: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Mengunduh '{MODEL_ID}' -> '{TARGET_DIR}' ...")
    snapshot_download(MODEL_ID, local_dir=TARGET_DIR, allow_patterns=ALLOW)
    print(f"[SUKSES] Model MT siap di '{TARGET_DIR}/'.")


if __name__ == "__main__":
    main()
