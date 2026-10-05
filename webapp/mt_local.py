"""
Backend terjemahan lokal (self-hosted) memakai MarianMT.

Menggantikan panggilan Google Translate (translate.googleapis.com) yang dari IP
VPS ini selalu dibalas HTTP 429 Too Many Requests.

Model default: Helsinki-NLP/opus-mt-mul-en (banyak bahasa -> Inggris).
Model dimuat lazy (hanya saat pertama dipakai) dan dijaga thread-lock karena
FastAPI melayani permintaan secara konkuren.
"""
import os
import threading
from typing import List, Optional

MT_MODEL_PATH = os.getenv("MT_MODEL_PATH", "/app/mt_model")
MT_LOCAL_ENABLED = os.getenv("MT_LOCAL_ENABLED", "1").lower() not in ("0", "false", "no", "")
MT_MAX_NEW_TOKENS = int(os.getenv("MT_MAX_NEW_TOKENS", "64"))
MT_MAX_INPUT_TOKENS = int(os.getenv("MT_MAX_INPUT_TOKENS", "192"))
MT_BATCH_SIZE = int(os.getenv("MT_BATCH_SIZE", "8"))

_lock = threading.Lock()
_tok = None
_mdl = None
_load_failed = False
_load_attempted = False


def local_available() -> bool:
    """True bila backend lokal diaktifkan dan direktori model ada."""
    return MT_LOCAL_ENABLED and os.path.isdir(MT_MODEL_PATH)


def _ensure_loaded() -> bool:
    global _tok, _mdl, _load_failed, _load_attempted
    if _mdl is not None:
        return True
    if _load_failed or not local_available():
        return False
    with _lock:
        if _mdl is not None:
            return True
        if _load_failed:
            return False
        _load_attempted = True
        try:
            from transformers import MarianMTModel, MarianTokenizer

            _tok = MarianTokenizer.from_pretrained(MT_MODEL_PATH)
            _mdl = MarianMTModel.from_pretrained(MT_MODEL_PATH)
            _mdl.eval()
            print(f"[info] MT lokal siap: {MT_MODEL_PATH}")
            return True
        except Exception as exc:  # noqa: BLE001
            _load_failed = True
            print(f"[peringatan] MT lokal gagal dimuat: {type(exc).__name__}: {exc}")
            return False


def translate_to_en(text: str) -> Optional[str]:
    """Terjemahkan teks ke Bahasa Inggris memakai model lokal.

    Mengembalikan None bila backend tidak tersedia atau terjadi kegagalan,
    sehingga pemanggil bisa jatuh ke fallback berikutnya.

    Batas token sengaja kecil (MT_MAX_INPUT_TOKENS=192, MT_MAX_NEW_TOKENS=64).
    Nilai lama 512/256 membuat KV cache MarianMT besar: satu teks dari val.csv
    (p90 = 264 kata) menaikkan peak RSS di atas batas container dan memicu
    OOM-kill -> 502. Klasifikasi juga hanya memakai MAX_LENGTH=192 token, jadi
    terjemahan yang lebih panjang itu tidak pernah dipakai.
    """
    text = (text or "").strip()
    if not text:
        return None
    if not _ensure_loaded():
        return None
    try:
        import torch

        with _lock, torch.inference_mode():
            batch = _tok(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=MT_MAX_INPUT_TOKENS
            )
            generated = _mdl.generate(**batch, max_new_tokens=MT_MAX_NEW_TOKENS)
            return _tok.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[peringatan] MT lokal gagal menerjemahkan: {type(exc).__name__}: {exc}")
        return None


def translate_many_to_en(texts: List[str]) -> List[Optional[str]]:
    """Terjemahkan banyak teks dalam SATU forward pass.

    `translate_to_en` dipanggil per teks, jadi 60 teks berarti 60 kali generate
    (~9 detik per teks pada CPU). Untuk evaluasi thousands baris itu tidak
    mungkin. Batch pendek (8) memotong wall-clock ~8x tanpa menambah peak
    memori berarti karena max_new_tokens dibatasi.

    Posisi yang gagal dikembalikan apa adanya (None) supaya pemanggil bisa
    jatuh ke fallback per teks.
    """
    out: List[Optional[str]] = [None] * len(texts)
    if not _ensure_loaded():
        return out
    try:
        import torch

        for i in range(0, len(texts), MT_BATCH_SIZE):
            window = texts[i:i + MT_BATCH_SIZE]
            keep = [j for j, t in enumerate(window) if t and t.strip()]
            if not keep:
                continue
            chunk = [window[j] for j in keep]
            with _lock, torch.inference_mode():
                batch = _tok(chunk, return_tensors="pt", padding=True,
                             truncation=True, max_length=MT_MAX_INPUT_TOKENS)
                generated = _mdl.generate(**batch, max_new_tokens=MT_MAX_NEW_TOKENS)
                decoded = _tok.batch_decode(generated, skip_special_tokens=True)
            for j, d in zip(keep, decoded):
                out[i + j] = d.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[peringatan] MT lokal batch gagal: {type(exc).__name__}: {exc}")
    return out


def status() -> dict:
    """Ringkasan status untuk endpoint /health."""
    return {
        "enabled": MT_LOCAL_ENABLED,
        "model_path": MT_MODEL_PATH,
        "dir_exists": os.path.isdir(MT_MODEL_PATH),
        "loaded": _mdl is not None,
        "load_failed": _load_failed,
        "load_attempted": _load_attempted,
    }
