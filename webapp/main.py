"""
Aplikasi web deteksi prompt injection (klasifikasi teks biner) berbasis
DeBERTa-v3-base hasil fine-tuning.

Fitur:
  - Cek satu teks               : POST /predict        {text}
  - Cek banyak teks sekaligus   : POST /predict_batch  {texts: [...]}
  - Unggah berkas .txt/.csv     : POST /predict_file   (multipart, satu teks per baris)
  - Status layanan              : GET  /health
  - Dokumentasi API otomatis    : GET  /docs

Menjalankan:
    pip install -r requirements.txt
    set MODEL_PATH=../best_model_biner      (Windows CMD)
    # export MODEL_PATH=../best_model_biner  (Linux/Mac)
    uvicorn main:app --host 0.0.0.0 --port 8000
Buka http://127.0.0.1:8000
"""
import io
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "../best_model_biner")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "192"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_LABELS = {0: "safe", 1: "prompt injection"}

# ---------------------------------------------------------------------------
# Preprocessing (mengikuti preprocess_opts.json agar identik dengan pelatihan)
# ---------------------------------------------------------------------------
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_HOMOGLYPH = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ԁ": "d", "ο": "o", "α": "a", "А": "A", "Е": "E", "О": "O", "С": "C",
}
_OPTS = {}


def _load_preprocess_opts(path):
    default = {"case_folding": True, "unicode_nfkc": True, "strip_zero_width": True,
               "normalize_whitespace": True, "homoglyph": True, "leetspeak": False}
    f = Path(path) / "preprocess_opts.json"
    if f.exists():
        try:
            default.update(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return default


def preprocess(text: str) -> str:
    if _OPTS.get("unicode_nfkc", True):
        text = unicodedata.normalize("NFKC", text)
    if _OPTS.get("homoglyph", True):
        text = text.translate({ord(k): v for k, v in _HOMOGLYPH.items()})
    if _OPTS.get("strip_zero_width", True):
        text = text.translate(_ZERO_WIDTH)
    if _OPTS.get("case_folding", True):
        text = text.casefold()
    if _OPTS.get("normalize_whitespace", True):
        text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# App & model
# ---------------------------------------------------------------------------
app = FastAPI(title="Deteksi Prompt Injection — DeBERTa-v3-base", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_tokenizer = None
_labels = DEFAULT_LABELS
_num_params = 0
_model_name = "DeBERTa-v3-base"


def load_model():
    global _model, _tokenizer, _labels, _OPTS, _num_params, _model_name
    if _model is not None:
        return
    _OPTS = _load_preprocess_opts(MODEL_PATH)
    print(f"[info] Memuat model dari '{MODEL_PATH}' ke {DEVICE} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.to(DEVICE).eval()
    id2label = getattr(_model.config, "id2label", None)
    if id2label and not all(str(v).lower().startswith("label_") for v in id2label.values()):
        _labels = {int(k): str(v).replace("_", " ") for k, v in id2label.items()}
    _num_params = sum(p.numel() for p in _model.parameters())
    base = (getattr(_model.config, "_name_or_path", "") or "").replace("\\", "/")
    if "/" in base:
        base = base.split("/")[-1]
    # tampilkan nama arsitektur yang rapi, bukan nama folder model lokal
    _model_name = base if "deberta" in base.lower() else "DeBERTa-v3-base"
    print(f"[info] Siap. Perangkat={DEVICE} Label={_labels} Opsi praproses={_OPTS}")


@app.on_event("startup")
def _startup():
    try:
        load_model()
    except Exception as exc:  # noqa: BLE001
        print("\n[peringatan] Gagal memuat model:", exc)
        print("[peringatan] Setel MODEL_PATH ke folder model biner, lalu jalankan ulang.\n")


# ---------------------------------------------------------------------------
# Skema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str] = Field(default_factory=list)


class PredictResult(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict


# ---------------------------------------------------------------------------
# Inti inferensi (batched)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def classify_many(texts: List[str]) -> List[PredictResult]:
    if _model is None:
        raise HTTPException(503, "Model belum termuat. Periksa MODEL_PATH lalu restart server.")
    clean = [preprocess(t) for t in texts]
    out: List[PredictResult] = []
    for i in range(0, len(clean), 64):  # sub-batch agar hemat memori
        chunk = clean[i:i + 64]
        enc = _tokenizer(chunk, truncation=True, max_length=MAX_LENGTH,
                         padding=True, return_tensors="pt").to(DEVICE)
        probs = F.softmax(_model(**enc).logits, dim=-1)
        for j, row in enumerate(probs):
            pid = int(torch.argmax(row).item())
            out.append(PredictResult(
                text=texts[i + j],
                label=_labels.get(pid, f"kelas {pid}"),
                confidence=round(float(row[pid]), 4),
                probabilities={_labels.get(k, f"kelas {k}"): round(float(p), 4)
                               for k, p in enumerate(row)},
            ))
    return out


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResult)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Teks masukan kosong.")
    return classify_many([text])[0]


@app.post("/predict_batch", response_model=List[PredictResult])
def predict_batch(req: BatchRequest):
    texts = [t for t in (req.texts or []) if t and t.strip()]
    if not texts:
        raise HTTPException(400, "Daftar teks kosong.")
    if len(texts) > MAX_BATCH:
        raise HTTPException(413, f"Maksimum {MAX_BATCH} teks per permintaan.")
    return classify_many(texts)


@app.post("/predict_file", response_model=List[PredictResult])
async def predict_file(file: UploadFile = File(...)):
    raw = (await file.read()).decode("utf-8", errors="replace")
    if file.filename and file.filename.lower().endswith(".csv"):
        import csv
        rows = list(csv.reader(io.StringIO(raw)))
        texts = [r[0] for r in rows if r and r[0].strip()]
        if texts and texts[0].lower() in ("text", "teks", "prompt"):
            texts = texts[1:]  # buang header
    else:
        texts = [ln for ln in raw.splitlines() if ln.strip()]
    if not texts:
        raise HTTPException(400, "Berkas tidak memuat teks yang bisa diproses.")
    if len(texts) > MAX_BATCH:
        raise HTTPException(413, f"Maksimum {MAX_BATCH} baris per berkas.")
    return classify_many(texts)


def preprocess_trace(text: str):
    """Jejak praproses per langkah (urutannya identik dengan preprocess())."""
    steps = []
    cur = text

    def add(name, new):
        nonlocal cur
        steps.append({"name": name, "before": cur, "after": new, "changed": new != cur})
        cur = new

    if _OPTS.get("unicode_nfkc", True):
        add("Normalisasi Unicode (NFKC)", unicodedata.normalize("NFKC", cur))
    if _OPTS.get("homoglyph", True):
        add("Homoglyph → Latin", cur.translate({ord(k): v for k, v in _HOMOGLYPH.items()}))
    if _OPTS.get("strip_zero_width", True):
        add("Hapus karakter zero-width", cur.translate(_ZERO_WIDTH))
    if _OPTS.get("case_folding", True):
        add("Case folding (huruf kecil)", cur.casefold())
    if _OPTS.get("normalize_whitespace", True):
        add("Rapikan spasi", re.sub(r"\s+", " ", cur).strip())
    return steps, cur


OCCLUSION_MAX_WORDS = int(os.getenv("OCCLUSION_MAX_WORDS", "60"))


def _run(texts, want_attn=False):
    """Forward pass batched. Kembalikan logits, probs, enc, output."""
    enc = _tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                     padding=True, return_tensors="pt").to(DEVICE)
    out = _model(**enc, output_attentions=want_attn)
    return out.logits, F.softmax(out.logits, dim=-1), enc, out


def _pred_dict(logits_row, probs_row):
    pid = int(torch.argmax(probs_row).item())
    return {
        "label": _labels.get(pid, f"kelas {pid}"),
        "confidence": round(float(probs_row[pid]), 4),
        "probabilities": {_labels.get(k, f"kelas {k}"): round(float(v), 4)
                          for k, v in enumerate(probs_row)},
    }


@app.post("/analyze")
@torch.inference_mode()
def analyze(req: PredictRequest):
    """Seperti /predict + jejak pipeline, occlusion importance, attention, dan
    perbandingan prediksi teks mentah vs setelah praproses."""
    if _model is None:
        raise HTTPException(503, "Model belum termuat. Periksa MODEL_PATH lalu restart server.")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Teks masukan kosong.")

    steps, clean = preprocess_trace(text)

    # Tokenisasi penuh (tanpa truncation) untuk hitung token sebenarnya
    ids_full = _tokenizer(clean, add_special_tokens=True, truncation=False)["input_ids"]
    tokens = _tokenizer.convert_ids_to_tokens(ids_full)
    truncated = len(ids_full) > MAX_LENGTH

    # --- Inferensi teks bersih (+ attention lapisan terakhir) ---
    logits_c, probs_c, enc_c, out_c = _run([clean], want_attn=True)
    logits1, probs1 = logits_c[0], probs_c[0]
    pid = int(torch.argmax(probs1).item())
    other = (1 - pid) if probs1.numel() == 2 else pid

    # --- #3 Prediksi teks mentah vs bersih ---
    logits_r, probs_r, _, _ = _run([text])
    pred_raw = _pred_dict(logits_r[0], probs_r[0])
    pred_clean = _pred_dict(logits1, probs1)

    # --- #2 Attention [CLS] -> token (rata-rata head, lapisan terakhir) ---
    attention = []
    try:
        attn = out_c.attentions[-1][0]              # (heads, seq, seq)
        cls_attn = attn[:, 0, :].mean(0).tolist()   # perhatian dari [CLS]
        enc_tokens = _tokenizer.convert_ids_to_tokens(enc_c["input_ids"][0].tolist())
        mx = max(cls_attn) or 1.0
        attention = [{"token": t, "weight": round(w, 4), "norm": round(w / mx, 4)}
                     for t, w in zip(enc_tokens, cls_attn)]
    except Exception:  # noqa: BLE001
        attention = []

    # --- #1 Occlusion importance level-kata (pakai margin logit agar tak jenuh) ---
    words = clean.split(" ") if clean else []
    importance = []
    occlusion_skipped = False
    if 1 <= len(words) <= OCCLUSION_MAX_WORDS:
        base_margin = float(logits1[pid] - logits1[other])
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        lo, _, _, _ = _run(variants)
        for i, w in enumerate(words):
            m = float(lo[i][pid] - lo[i][other])
            importance.append({"word": w, "delta": round(base_margin - m, 4)})
    elif len(words) > OCCLUSION_MAX_WORDS:
        occlusion_skipped = True

    return {
        "text": text,
        "preprocessed": clean,
        "steps": steps,
        "tokens": tokens,
        "token_ids": ids_full,
        "num_tokens": len(ids_full),
        "max_length": MAX_LENGTH,
        "truncated": truncated,
        "logits": {_labels.get(k, f"kelas {k}"): round(float(v), 4) for k, v in enumerate(logits1)},
        "probabilities": pred_clean["probabilities"],
        "label": pred_clean["label"],
        "confidence": pred_clean["confidence"],
        "pred_raw": pred_raw,
        "pred_clean": pred_clean,
        "preproc_changed_prediction": pred_raw["label"] != pred_clean["label"],
        "attention": attention,
        "importance": importance,
        "occlusion_skipped": occlusion_skipped,
        "occlusion_max_words": OCCLUSION_MAX_WORDS,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None,
            "device": DEVICE, "labels": _labels, "max_length": MAX_LENGTH,
            "parameters": _num_params, "model_name": _model_name}


# ---------------------------------------------------------------------------
# Frontend statis
# ---------------------------------------------------------------------------
_STATIC = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
