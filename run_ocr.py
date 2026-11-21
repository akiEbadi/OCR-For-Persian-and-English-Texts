# Step 2: run Tesseract and PaddleOCR on all test images.
# Output: results/predictions.csv

import csv
import os
import time

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from paddleocr import TextRecognition

# ---------------- Tesseract settings ----------------
TESS_DATA = os.path.abspath("models/tessdata_best")   # folder with fas/eng .traineddata
TESS_LANG = {"fa": "fas", "en": "eng"}
TESS_PSM = 7        # 7 = the image is one line of text
TESS_OEM = 1        # 1 = LSTM (neural network) engine
# Tesseract 5 sometimes throws away Persian lines as "noise"; these two turn that off
TESS_EXTRA = "-c textord_noise_rejrows=0 -c textord_noise_rejwords=0"
TESS_BORDER = 10    # border added before Tesseract (it reads better with some space)

# ---------------- PaddleOCR settings ----------------
PADDLE_MODEL = {
    "fa": "arabic_PP-OCRv5_mobile_rec",   # Arabic-script model, also covers Persian
    "en": "PP-OCRv6_medium_rec",
}


def tesseract_config(psm=TESS_PSM, extra=TESS_EXTRA):
    return f"--tessdata-dir {TESS_DATA} --oem {TESS_OEM} --psm {psm} {extra}"


def run_tesseract(path, lang, psm=TESS_PSM, extra=TESS_EXTRA, border=TESS_BORDER):
    img = Image.open(path).convert("L")
    if border > 0:
        # same color as the paper; a pure white frame on a gray photo is read as "[" or "|"
        background = int(np.median(np.array(img)))
        img = ImageOps.expand(img, border=border, fill=background)
    text = pytesseract.image_to_string(img, lang=TESS_LANG[lang], config=tesseract_config(psm, extra))
    return text.strip()


def load_paddle_models():
    return {lang: TextRecognition(model_name=name) for lang, name in PADDLE_MODEL.items()}


def run_paddle(model, path):
    for res in model.predict(path, batch_size=1):
        return res["rec_text"]
    return ""


def load_rows():
    # made images
    with open("data/labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # lines cut from our own page photos - only the ones whose text is typed in the Excel file
    if os.path.exists("data/real_lines.xlsx"):
        real = pd.read_excel("data/real_lines.xlsx", dtype=str).fillna("")
        for _, r in real.iterrows():
            if r["text"].strip():
                rows.append({"file": r["file"], "lang": r["lang"], "font": "photo",
                             "condition": "photo", "text": r["text"].strip()})
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    rows = load_rows()

    print("loading PaddleOCR models ...")
    paddle = load_paddle_models()

    out = []
    for n, row in enumerate(rows, start=1):
        path = os.path.join("images", row["file"])
        lang = row["lang"]

        t0 = time.time()
        tess_text = run_tesseract(path, lang)
        t1 = time.time()
        paddle_text = run_paddle(paddle[lang], path)
        t2 = time.time()

        out.append({
            **row,
            "tesseract": tess_text,
            "paddle": paddle_text,
            "tesseract_time": round(t1 - t0, 4),
            "paddle_time": round(t2 - t1, 4),
        })
        if n % 20 == 0:
            print(f"{n}/{len(rows)} done")

    with open("results/predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print("saved results/predictions.csv")


if __name__ == "__main__":
    main()
