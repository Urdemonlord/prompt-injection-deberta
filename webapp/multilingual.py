"""
Modul Lapisan Adaptasi Multibahasa (Multilingual Adaptation Layer).

Menyediakan deteksi bahasa otomatis dan pemetaan semantik ke Bahasa Inggris
sebelum teks diproses oleh pipeline praproses dan model DeBERTa-v3-base.

Terjemahan hanya dijalankan untuk teks yang terdeteksi Bahasa Indonesia.
Model MT adalah many-to-one tanpa token kode bahasa, jadi menerjemahkan teks
yang sudah Inggris hanya menghasilkan tebakan (mis. "hai" -> "yes").
"""
import re
from typing import Dict, List, Tuple

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


def translate_text(text: str) -> Tuple[str, str]:
    """Terjemahkan teks Bahasa Indonesia ke Inggris memakai model lokal.

    Mengembalikan (teks_terjemahan, status). Teks yang gagal diterjemahkan
    dikembalikan apa adanya dengan status `passthrough`, supaya pemanggil bisa
    lanjut ke klasifikasi tanpa menebak.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned, "empty"

    # --- Backend terjemahan lokal (self-hosted) ---
    if mt_local is not None and mt_local.local_available():
        translated = mt_local.translate_to_en(cleaned)
        if translated:
            if translated.lower() != cleaned.lower():
                return translated, "local_mt"
            return cleaned, "local_mt_passthrough"

    return cleaned, "passthrough"


def adapt_multilingual(text: str, enabled: bool = True) -> Dict:
    """
    Lapisan Adaptasi Multibahasa utama.
    Jika enabled=True dan teks terdeteksi Bahasa Indonesia, teks diterjemahkan
    ke Bahasa Inggris.
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

    # Gate bahasa SEBELUM memanggil MT. Tanpa gate ini, `opus-mt-mul-en`
    # (many-to-one, tanpa token kode bahasa) menebak bahasa sumber dari teks
    # dan menerjemahkan teks Inggris yang sudah jadi — "hai" -> "yes".
    lang_code = _heuristic_detect(original)
    if lang_code != "id":
        return {
            "original_text": original,
            "adapted_text": original,
            "detected_lang": lang_code,
            "lang_name": LANG_MAP.get(lang_code, f"🌐 {lang_code.upper()}"),
            "is_translated": False,
            "status": "passthrough",
        }

    translated, status = translate_text(original)
    lang_name = LANG_MAP.get(lang_code, f"🌐 {lang_code.upper()}")
    is_translated = status == "local_mt" and translated.lower() != original.lower()

    return {
        "original_text": original,
        "adapted_text": translated if is_translated else original,
        "detected_lang": lang_code,
        "lang_name": lang_name,
        "is_translated": is_translated,
        "status": status if is_translated else "passthrough",
    }


def adapt_multilingual_many(texts: List[str], enabled: bool = True) -> List[Dict]:
    """Versi batch dari adapt_multilingual.

    Melewati jalur terjemahan untuk teks yang sudah Inggris (heuristik) — MarianMT
    untuk teks EN menghasilkan passthrough yang sia-sia dan mahal. Hanya teks
    ID yang diterjemahkan, dalam satu panggilan batch.
    """
    if not enabled:
        return [adapt_multilingual(t, enabled=False) for t in texts]

    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    result = [adapt_multilingual("", enabled=False) for _ in texts]
    need = [i for i in idx if _heuristic_detect(texts[i]) == "id"]

    translated_map = {}
    if need and mt_local is not None and mt_local.local_available():
        outs = mt_local.translate_many_to_en([texts[i] for i in need])
        translated_map = {i: o for i, o in zip(need, outs) if o}

    for i in idx:
        lang_code = _heuristic_detect(texts[i])
        tr = translated_map.get(i)
        original = texts[i].strip()
        if not tr or tr.lower() == original.lower():
            result[i] = {
                "original_text": original,
                "adapted_text": original,
                "detected_lang": lang_code,
                "lang_name": LANG_MAP.get(lang_code, f"\U0001F310 {lang_code.upper()}"),
                "is_translated": False,
                "status": "local_mt_passthrough" if tr else "passthrough",
            }
            continue
        result[i] = {
            "original_text": original,
            "adapted_text": tr,
            "detected_lang": lang_code,
            "lang_name": LANG_MAP.get(lang_code, f"\U0001F310 {lang_code.upper()}"),
            "is_translated": True,
            "status": "local_mt",
        }
    return result
