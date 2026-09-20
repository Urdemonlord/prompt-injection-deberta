---
title: Deteksi Prompt Injection (DeBERTa-v3)
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Deteksi Prompt Injection — DeBERTa-v3-base (biner)

Aplikasi web klasifikasi teks biner (**safe** / **prompt injection**) berbasis
DeBERTa-v3-base hasil fine-tuning. Backend **FastAPI**, frontend HTML/CSS/JS.

## Cara pakai
- Buka halaman utama untuk cek satu teks atau batch (drag-and-drop `.txt`/`.csv`).
- Dokumentasi API otomatis di `/docs`.

## Endpoint
| Metode | Path | Fungsi |
|---|---|---|
| POST | `/predict` | `{ "text": "..." }` → satu hasil |
| POST | `/predict_batch` | `{ "texts": ["...","..."] }` → daftar hasil |
| POST | `/predict_file` | multipart `.txt`/`.csv` → daftar hasil |
| GET  | `/health` | status model & perangkat |

Model dimuat dari folder `best_model_biner/` (hasil `save_pretrained()`),
dengan praproses identik pelatihan (baca `preprocess_opts.json`).
