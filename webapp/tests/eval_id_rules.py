#!/usr/bin/env python3
"""Bandingkan aturan keputusan pada eval set Bahasa Indonesia berlabel manual.

Menghitung metrik untuk 4 aturan (union / raw / translated / AND) memakai
skor yang benar-benar dikembalikan model, bukan label ground truth.

Sumber: eval_id_safe.json (93 teks aman) + eval_id_injection.json (34 adversarial).
Jalankan: python3 eval_id_rules.py [--mt on|off]
"""
import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8002"


def load():
    rows = []
    for name in ("eval_id_safe.json", "eval_id_injection.json"):
        path = os.path.join(HERE, name)
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return rows


def inj(label):
    return "inject" in (label or "").lower()


def predict(texts, multilingual):
    body = json.dumps({"texts": texts, "multilingual": multilingual}).encode()
    req = urllib.request.Request(URL + "/predict_batch", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def metrics(rs, decider):
    tp = fp = tn = fn = 0
    for r in rs:
        pred = decider(r["raw"], r["tr"])
        if r["exp"]:
            tp, fn = (tp + 1, fn) if pred else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if pred else (fp, tn + 1)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return tp, fp, tn, fn, p, rc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", choices=["on", "off"], default="on")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    rows = load()
    texts = [r["text"] for r in rows]
    print(f"eval set: {len(rows)} teks "
          f"({sum(1 for r in rows if r['label'] == 'prompt injection')} adversarial, "
          f"{sum(1 for r in rows if r['label'] == 'safe')} aman), MT={args.mt}",
          file=sys.stderr)

    out = []
    CHUNK = 32
    for i in range(0, len(texts), CHUNK):
        out.extend(predict(texts[i:i + CHUNK], args.mt == "on"))
        print(f"  {min(i + CHUNK, len(texts))}/{len(texts)}", file=sys.stderr)
    assert len(out) == len(texts)

    recs = []
    for gold, pred in zip(rows, out):
        u = (pred.get("multilingual_info") or {}).get("union") or {}
        has_tr = u.get("translated_label") is not None
        recs.append({
            "id": gold["id"], "cat": gold["category"], "exp": inj(gold["label"]),
            # fallback ke row["label"] API = prediksi model, bukan ground truth
            "raw": inj(u.get("raw_label") or pred.get("label")),
            "tr": inj(u.get("translated_label")) if has_tr else None,
            "conf": pred.get("confidence", 0.0),
            "text": gold["text"],
        })

    rules = {
        "union (sekarang)": lambda a, b: a or b,
        "raw saja": lambda a, b: a,
        "translated saja": lambda a, b: bool(b),
        "AND (tr menang)": lambda a, b: bool(b),
    }

    print(f"\n=== MT={args.mt} ===")
    print(f"{'aturan':20s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} "
          f"{'prec':>6s} {'rec':>6s} {'F1':>6s}")
    best = ("-", 0.0)
    for name, fn in rules.items():
        tp, fp, tn, fn_, p, rc, f1 = metrics(recs, fn)
        print(f"{name:20s} {tp:4d} {fp:4d} {tn:4d} {fn_:4d} "
              f"{p:6.3f} {rc:6.3f} {f1:6.3f}")
        if best is None or f1 > best[1]:
            best = (name, f1)
    print(f"\nterbaik: {best[0]} (F1={best[1]:.3f})")

    # Disagreements: kasus di mana aturan berbeda -- ini yang butuh review manusia.
    dis = [r for r in recs if r["tr"] is not None and r["raw"] != r["tr"]]
    print(f"\nraw != translated: {len(dis)}/{len(recs)}")
    for tag, keep in (("raw=INJ tr=safe (union bikin FP)", lambda r: r["raw"] and not r["tr"]),
                      ("raw=safe tr=INJ (union menyelamatkan)", lambda r: not r["raw"] and r["tr"])):
        sel = [r for r in dis if keep(r)]
        print(f"  {tag}: {len(sel)}")
        for r in sel[:6]:
            print(f"    [{r['id']}/{r['cat']}] conf={r['conf']:.3f} {r['text'][:70]}")

    with open("/tmp/eval_id_rules.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    print("\ntersimpan: /tmp/eval_id_rules.json")


if __name__ == "__main__":
    main()