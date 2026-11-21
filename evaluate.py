# Step 3: compare the outputs with the correct text.
# CER = character error rate, WER = word error rate
# Output: results/summary.csv

import jiwer
import pandas as pd

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PUNCTUATION = ".,،؛;:!?؟"


def normalize(text):
    text = str(text)
    # Arabic letters that look like Persian ones
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    # Persian / Arabic digits -> 0-9
    for i in range(10):
        text = text.replace(PERSIAN_DIGITS[i], str(i)).replace(ARABIC_DIGITS[i], str(i))
    # half-space (ZWNJ): Tesseract keeps it, PaddleOCR writes a normal space.
    # we count both as a space so this difference is not an error
    text = text.replace("\u200c", " ")
    # remove extra spaces
    return " ".join(text.split())


def no_punct(text):
    for ch in PUNCTUATION:
        text = text.replace(ch, "")
    return " ".join(text.split())


def scores(refs, hyps, ignore_punct=False):
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    if ignore_punct:
        refs = [no_punct(r) for r in refs]
        hyps = [no_punct(h) for h in hyps]
    exact = sum(r == h for r, h in zip(refs, hyps)) / len(refs)
    return jiwer.cer(refs, hyps) * 100, jiwer.wer(refs, hyps) * 100, exact * 100


def main():
    df = pd.read_csv("results/predictions.csv", keep_default_na=False)

    rows = []
    for lang in ["fa", "en"]:
        for condition in ["clean", "noisy", "clean+noisy", "photo"]:
            part = df[df["lang"] == lang]
            if condition == "clean+noisy":
                part = part[part["condition"].isin(["clean", "noisy"])]
            else:
                part = part[part["condition"] == condition]
            if len(part) == 0:          # e.g. no photo lines typed yet
                continue
            for tool in ["tesseract", "paddle"]:
                cer, wer, exact = scores(part["text"], part[tool])
                cer2, _, _ = scores(part["text"], part[tool], ignore_punct=True)
                rows.append({
                    "lang": lang,
                    "condition": condition,
                    "tool": tool,
                    "CER %": round(cer, 2),
                    "WER %": round(wer, 2),
                    "exact %": round(exact, 1),
                    "CER no punct %": round(cer2, 2),
                    "ms per line": round(part[tool + "_time"].mean() * 1000, 1),
                })

    summary = pd.DataFrame(rows)
    summary.to_csv("results/summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
