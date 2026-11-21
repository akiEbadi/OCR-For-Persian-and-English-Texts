# Step 1: make test images from the sentence files.
# Each sentence is written with 2 fonts, once clean and once noisy.
# Output: images/*.png and data/labels.csv

import csv
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS = {
    "fa": {
        "vazir": "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf",
        "nazli": "/usr/share/fonts/truetype/farsiweb/nazli.ttf",
    },
    "en": {
        "liberation": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    },
}
FONT_SIZE = 32
MARGIN = 4  # small white margin around the text


def draw_line(text, font_path):
    font = ImageFont.truetype(font_path, FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    left, top, right, bottom = font.getbbox(text)
    width = right - left + 2 * MARGIN
    height = bottom - top + 2 * MARGIN
    img = Image.new("L", (width, height), 255)  # white grayscale image
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN - left, MARGIN - top), text, font=font, fill=0)  # black text
    return img


def make_noisy(img):
    # a little blur + random noise, like a bad scan
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.array(img).astype(float)
    arr = arr + np.random.normal(0, 25, arr.shape)
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    random.seed(0)
    np.random.seed(0)
    os.makedirs("images", exist_ok=True)

    rows = []
    for lang in ["fa", "en"]:
        with open(f"data/sentences_{lang}.txt", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        for i, text in enumerate(sentences, start=1):
            for font_name, font_path in FONTS[lang].items():
                clean = draw_line(text, font_path)
                noisy = make_noisy(clean)
                for condition, img in [("clean", clean), ("noisy", noisy)]:
                    file_name = f"{lang}_{i:02d}_{font_name}_{condition}.png"
                    img.save(os.path.join("images", file_name))
                    rows.append([file_name, lang, font_name, condition, text])

    with open("data/labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "lang", "font", "condition", "text"])
        writer.writerows(rows)

    print("images made:", len(rows))


if __name__ == "__main__":
    main()
