# Step 4: see how some inference settings change the result.
# Uses the noisy images only (where settings matter more).
# Output: results/params.csv

import csv
import os
import time

import pandas as pd
from PIL import Image, ImageOps

from evaluate import scores
from run_ocr import load_paddle_models, run_paddle, run_tesseract


def main():
    with open("data/labels.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["condition"] == "noisy"]
    paddle = load_paddle_models()
    out = []

    def add(tool, setting, lang, refs, hyps, seconds):
        cer, wer, _ = scores(refs, hyps)
        out.append({"tool": tool, "setting": setting, "lang": lang,
                    "CER %": round(cer, 2), "WER %": round(wer, 2),
                    "ms per line": round(seconds / len(refs) * 1000, 1)})
        print(out[-1])

    for lang in ["fa", "en"]:
        part = [r for r in rows if r["lang"] == lang]
        paths = [os.path.join("images", r["file"]) for r in part]
        refs = [r["text"] for r in part]

        # --- Tesseract: page segmentation mode (--psm) ---
        for psm in [6, 7, 13]:
            t0 = time.time()
            hyps = [run_tesseract(p, lang, psm=psm) for p in paths]
            add("tesseract", f"psm {psm}", lang, refs, hyps, time.time() - t0)

        # --- Tesseract: default noise filter (on) vs. our setting (off) ---
        t0 = time.time()
        hyps = [run_tesseract(p, lang, extra="") for p in paths]
        add("tesseract", "noise filter on (default)", lang, refs, hyps, time.time() - t0)

        # --- Tesseract: no white border ---
        t0 = time.time()
        hyps = [run_tesseract(p, lang, border=0) for p in paths]
        add("tesseract", "no border", lang, refs, hyps, time.time() - t0)

        # --- PaddleOCR: image as it is ---
        t0 = time.time()
        hyps = [run_paddle(paddle[lang], p) for p in paths]
        add("paddle", "tight crop (normal)", lang, refs, hyps, time.time() - t0)

        # --- PaddleOCR: with a big white border (shows it prefers tight crops) ---
        os.makedirs("results/tmp", exist_ok=True)
        padded = []
        for p in paths:
            q = os.path.join("results/tmp", os.path.basename(p))
            ImageOps.expand(Image.open(p), border=30, fill=255).save(q)
            padded.append(q)
        t0 = time.time()
        hyps = [run_paddle(paddle[lang], q) for q in padded]
        add("paddle", "30 px white border", lang, refs, hyps, time.time() - t0)

        # --- PaddleOCR: batch size (speed only) ---
        t0 = time.time()
        hyps = [res["rec_text"] for res in paddle[lang].predict(paths, batch_size=8)]
        add("paddle", "batch size 8", lang, refs, hyps, time.time() - t0)

    pd.DataFrame(out).to_csv("results/params.csv", index=False)
    print("saved results/params.csv")


if __name__ == "__main__":
    main()
