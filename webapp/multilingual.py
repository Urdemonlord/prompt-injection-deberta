"""
Modul Lapisan Adaptasi Multibahasa (Multilingual Adaptation Layer).

Menyediakan deteksi bahasa otomatis dan pemetaan semantik ke Bahasa Inggris
sebelum teks diproses oleh pipeline praproses dan model DeBERTa-v3-base.
Dilengkapi graceful offline fallback jika tidak ada koneksi internet.
"""
import json
import re
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

try:
    import mt_local
except Exception:  # noqa: BLE001
    mt_local = None

# Pemetaan kode bahasa ke nama & bendera representasi
LANG_MAP = {
    "id": "🇮🇩 Bahasa Indonesia",
    "en": "🇬🇧 English",
    "jv": "🇮🇩 Bahasa Jawa",
    "su": "🇮🇩 Bahasa Sunda",
    "ms": "🇲🇾 Bahasa Melayu",
    "es": "🇪🇸 Español",
    "fr": "🇫🇷 Français",
    "de": "🇩🇪 Deutsch",
    "zh-CN": "🇨🇳 中文 (Sederhana)",
    "zh-TW": "🇨🇳 中文 (Tradisional)",
    "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский",
    "pt": "🇵🇹 Português",
    "it": "🇮🇹 Italiano",
    "vi": "🇻🇳 Tiếng Việt",
    "th": "🇹🇭 ไทย",
}

# Heuristik cepat kata-kata umum Bahasa Indonesia untuk deteksi lokal/offline
_ID_KEYWORDS = {
    "dan", "yang", "untuk", "di", "dari", "ini", "itu", "kamu", "saya", "aku",
    "tolong", "abaikan", "jangan", "adalah", "bisa", "buatkan", "tampilkan",
    "perintah", "aturan", "sebelumnya", "sekarang", "menjadi", "bagaimana",
    "dengan", "pada", "ke", "karena", "agar", "tidak", "bukan", "hanya",
    "semua", "mereka", "kita", "anda", "berikan", "rahasia", "larangan"
}


def _heuristic_detect(text: str) -> str:
    """Deteksi cepat berbasis frekuensi kata kunci untuk fallback offline."""
    words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    if not words:
        return "en"
    id_matches = words.intersection(_ID_KEYWORDS)
    if len(id_matches) >= 2 or (len(words) <= 4 and len(id_matches) >= 1):
        return "id"
    return "en"


def translate_text(text: str, target_lang: str = "en", timeout_sec: float = 3.5) -> Tuple[str, str, str]:
    """
    Menerjemahkan teks ke target_lang dan mendeteksi bahasa asal.
    Mengembalikan (teks_terjemahan, kode_bahasa, status).

    Urutan backend:
      1. Model lokal (MarianMT)  -> status "local_mt"
      2. Google Translate publik -> status "adapted"
      3. Fallback heuristik      -> status "offline_fallback"
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned, "en", "empty"

    # --- 1) Backend terjemahan lokal (self-hosted) ---
    if mt_local is not None and mt_local.local_available():
        translated = mt_local.translate_to_en(cleaned)
        if translated:
            lang_code = _heuristic_detect(cleaned)
            if translated.lower() != cleaned.lower():
                return translated, lang_code, "local_mt"
            return cleaned, lang_code, "local_mt_passthrough"

    url = (
        "https://translate.googleapis.com/translate_a/single?"
        f"client=gtx&sl=auto&tl={target_lang}&dt=t&q="
        + urllib.parse.quote(cleaned)
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            chunks = [item[0] for item in data[0] if item and item[0]]
            translated = "".join(chunks).strip() if chunks else cleaned
            detected = data[2] if len(data) > 2 and isinstance(data[2], str) else "auto"
            return translated, detected, "adapted"
    except Exception:
        # Graceful fallback jika offline / timeout / limit
        fallback_lang = _heuristic_detect(cleaned)
        return cleaned, fallback_lang, "offline_fallback"


def adapt_multilingual(text: str, enabled: bool = True) -> Dict:
    """
    Lapisan Adaptasi Multibahasa utama.
    Jika enabled=True dan bahasa bukan 'en', teks diterjemahkan ke Bahasa Inggris.
    """
    original = text or ""
    if not enabled or not original.strip():
        return {
            "original_text": original,
            "adapted_text": original,
            "detected_lang": "en",
            "lang_name": LANG_MAP.get("en", "🇬🇧 English"),
            "is_translated": False,
            "status": "disabled" if not enabled else "empty",
        }

    translated, lang_code, status = translate_text(original, target_lang="en")
    lang_name = LANG_MAP.get(lang_code, f"🌐 {lang_code.upper()}")

    # Backend yang menandakan terjemahan benar-benar berhasil
    TRANSLATED_STATUSES = {"adapted", "local_mt"}

    # Bahasa sudah Inggris dan tidak ada terjemahan -> lewatkan apa adanya
    if lang_code == "en" and status not in TRANSLATED_STATUSES:
        return {
            "original_text": original,
            "adapted_text": original,
            "detected_lang": "en",
            "lang_name": LANG_MAP.get("en", "🇬🇧 English"),
            "is_translated": False,
            "status": "passthrough" if status.startswith("local_mt") else status,
        }

    is_translated = status in TRANSLATED_STATUSES and translated.lower() != original.lower()

    return {
        "original_text": original,
        "adapted_text": translated if is_translated else original,
        "detected_lang": lang_code,
        "lang_name": lang_name,
        "is_translated": is_translated,
        "status": status if is_translated else "passthrough",
    }
