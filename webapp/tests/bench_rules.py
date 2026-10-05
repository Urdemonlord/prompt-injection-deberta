#!/usr/bin/env python3
"""Bandingkan aturan keputusan (union / raw / translated) pada data berlabel asli.

Sumber: test.csv (56.473 baris held-out, nol overlap dengan val.csv).
PERINGATAN: jangan pakai val.csv untuk evaluasi model -- itu split yang dipakai
training, jadi F1 = 1.000 palsu (terukur: 2000/2000 benar).
Tujuan: putuskan aturan mana yang benar-benar menang, bukan berdasarkan kasus tulisan tangan.

Kenapa stratified + random: dataset balanced tapi hanya 99 baris ID-like, jadi
sampel acak bisa jatuh 100% EN. Dipisah laporan per bahasa.
"""
import argparse
import csv
import json
import random
import sys
import urllib.request

URL = "http://127.0.0.1:8002"
VAL = "/home/meowlabs/prompt-injection-deberta/test.csv"  # held-out, BUKAN val.csv

# Kata umum ID — untuk mengelompokkan laporan, bukan untuk memutuskan label.
ID_MARKERS = {
    "yang", "dan", "untuk", "di", "dari", "ini", "itu", "kamu", "saya", "aku",
    "tolong", "adalah", "bisa", "tidak", "bukan", "atau", "sudah", "akan",
    "dengan", "pada", "karena", "agar", "hanya", "semua", "mereka", "kita",
    "anda", "kalau", "atau", "jadi", "saja", "lagi", "masih", "harus",
}
ID_STOP = {"yang", "dan", "untuk", "di", "dari", "ini", "itu", "adalah", "bisa",
           "dengan", "pada", "atau", "saja", "lagi", "hanya", "semua", "itu"}


def looks_indonesian(text):
    """Bahasa ID bila ada >=2 kata umum ID yang bukan stopword."""
    words = [w.strip(".,!?;:'\"()").lower() for w in text.split()]
    hits = [w for w in words if w in ID_MARKERS and w not in ID_STOP]
    return len(hits) >= 2


def sample(path, n, seed=13):
    """Ambil n baris per kelas, acak tapi reproducible."""
    buckets = {"prompt_injection": [], "safe": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = buckets.get(row["label"])
            if b is not None:
                b.append(row["text_clean"])
    rng = random.Random(seed)
    out = []
    for label, rows in buckets.items():
        rng.shuffle(rows)
        out.extend((label, t) for t in rows[: n // 2])
    return out


def batch(texts, multilingual=True):
    """Kirim per chunk; server sub-batch INFER_BATCH internally."""
    body = json.dumps({"texts": texts, "multilingual": multilingual}).encode()
    req = urllib.request.Request(URL + "/predict_batch", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def inj(label):
    return "inject" in (label or "").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000, help="total baris sampel")
    ap.add_argument("--chunk", type=int, default=64, help="baris per request (max 64)")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--no-mt", action="store_true",
                    help="multilingual=false (tanpa jalur terjemahan)")
    args = ap.parse_args()

    pairs = sample(VAL, args.n)
    texts = [t for _, t in pairs]
    print(f"sampel: {len(texts)} baris dari {VAL} "
          f"(balanced per kelas, seed=13)", file=sys.stderr)

    # Chunk kecil WAJIB. Chunk 256 sudah terbukti OOM-kill: peak 1285 MB >
    # limit 1280 MB. Teks val.csv panjang (p90 = 264 kata) membuat generate
    # MarianMT menyimpan KV cache besar.
    rows = []
    CHUNK = args.chunk
    for i in range(0, len(texts), CHUNK):
        rows.extend(batch(texts[i:i + CHUNK], multilingual=not args.no_mt))
        print(f"  {min(i+CHUNK, len(texts))}/{len(texts)}", file=sys.stderr)
    assert len(rows) == len(texts)

    rules = {
        "union (sekarang)": lambda a, b: a or b,
        "raw saja":         lambda a, b: a,
        "translated saja":  lambda a, b: b,
        "AND (tr menang)":  lambda a, b: b,
    }

    # Kumpulkan metrics: global + per kelompok bahasa + per confidence tier.
    groups = {"all": [], "ID": [], "EN/other": []}
    recs = []
    for (exp_label, text), row in zip(pairs, rows):
        exp = inj(exp_label)
        u = (row.get("multilingual_info") or {}).get("union") or {}
        raw_l, tr_l = u.get("raw_label"), u.get("translated_label")
        # Tanpa union block = tidak ada terjemahan (passthrough / non-ID).
        # union block hanya ada bila adaptasi benar-benar dijalankan (teks ID).
        # Teks EN: tidak ada block -> translated bukan sinyal, jangan dihakimi FN.
        has_tr = u.get("applied") is True and u.get("translated_label") is not None
        raw_inj = inj(raw_l) if raw_l else inj(row["label"])
        tr_inj = inj(tr_l) if has_tr else None
        grp = "ID" if looks_indonesian(text) else "EN/other"
        recs.append({"exp": exp, "raw": raw_inj, "tr": tr_inj, "grp": grp,
                     "conf": row["confidence"], "text": text})
        groups["all"].append(recs[-1])
        groups[grp].append(recs[-1])

    def metrics(rs, decider):
        tp = fp = tn = fn = 0
        for r in rs:
            pred = decider(r["raw"], bool(r["tr"]))
            if r["exp"]:
                tp, fn = (tp + 1, fn) if pred else (tp, fn + 1)
            else:
                fp, tn = (fp + 1, tn) if pred else (fp, tn + 1)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return tp, fp, tn, fn, p, rc, f1

    print(f"\n=== multilingual={not args.no_mt} ===")
    for gname, rs in groups.items():
        if not rs:
            continue
        print(f"\n[{gname}] n={len(rs)}")
        print(f"  {'aturan':20s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} "
              f"{'prec':>6s} {'rec':>6s} {'F1':>6s}")
        for rname, dec in rules.items():
            tp, fp, tn, fn, p, rc, f1 = metrics(rs, dec)
            print(f"  {rname:20s} {tp:4d} {fp:4d} {tn:4d} {fn:4d} "
                  f"{p:6.3f} {rc:6.3f} {f1:6.3f}")

    # Disagreement = bahan diagnosis
    dis = [r for r in recs if r["tr"] is not None and r["raw"] != r["tr"]]
    print(f"\nraw != translated: {len(dis)}/{len(recs)} "
          f"({100*len(dis)/len(recs):.1f}%)")
    dis_inj = [r for r in dis if r["raw"] and not r["tr"]]
    dis_safe = [r for r in dis if not r["raw"] and r["tr"]]
    print(f"  raw=INJ tr=safe (union merusak FP): {len(dis_inj)}")
    print(f"  raw=safe tr=INJ (union menyelamatkan): {len(dis_safe)}")

    # Top disagreement by confidence — contoh paling informatif
    def show(title, rs):
        print(f"\n{title} (5 teratas)")
        for r in sorted(rs, key=lambda x: -x["conf"])[:5]:
            print(f"  conf={r['conf']:.3f} exp={'INJ' if r['exp'] else 'safe':4s} "
                  f"raw={'INJ' if r['raw'] else 'safe':4s} "
                  f"tr={'INJ' if r['tr'] else 'safe':4s} :: {r['text'][:66]}")

    show("FP yang dihapus translated-path", dis_inj)
    show("FN yang diselamatkan translated-path", dis_safe)

    out = {
        "n": len(recs),
        "multilingual": not args.no_mt,
        "metrics": {g: {r: metrics(rs, d) for r, d in rules.items()}
                    for g, rs in groups.items() if rs},
        "disagreement": {"total": len(dis), "raw_inj_tr_safe": len(dis_inj),
                         "raw_safe_tr_inj": len(dis_safe)},
    }
    with open("/tmp/rule_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\ntersimpan: /tmp/rule_bench.json")


if __name__ == "__main__":
    main()