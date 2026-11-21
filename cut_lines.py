# Cut real page photos into single text lines.
# Input : data/real_photos/fa_page.png, data/real_photos/en_page.jpg
# Output: images/real_fa_01.png ... and data/real_lines.xlsx
#         The Excel file shows each line image. Type the correct text of each line in column D.

import os

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font

PHOTOS = {"fa": "data/real_photos/fa_page.png", "en": "data/real_photos/en_page.jpg"}
MARGIN = 4      # white space left and right of a line
MARGIN_Y = 1    # small, so we do not catch bits of the lines above/below


def rotate(img, angle, border):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, m, (w, h), borderValue=border)


def find_skew(binary):
    # try small angles and keep the one where the text rows are sharpest
    best_angle, best_score = 0, 0
    for angle in np.arange(-5, 5.01, 0.25):
        score = np.var(rotate(binary, angle, 0).sum(axis=1))
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def cut_page(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # adaptive threshold: works even when part of the page is in shadow
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)

    # straighten the page
    angle = find_skew(binary)
    gray = rotate(gray, angle, 255)
    binary = rotate(binary, angle, 0)

    # rows with enough ink = text lines
    row_ink = (binary > 0).sum(axis=1)
    is_text = row_ink > 0.05 * row_ink.max()
    bands, start = [], None
    for y, t in enumerate(is_text):
        if t and start is None:
            start = y
        if not t and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(is_text)))

    median = np.median([b - a for a, b in bands if b - a > 3])
    lines, skipped = [], 0
    for a, b in bands:
        if b - a < 0.5 * median:      # dots, page number, dirt
            continue
        if b - a > 1.6 * median:      # several bent lines stuck together (page curve)
            skipped += 1
            continue
        cols = np.where((binary[a:b] > 0).sum(axis=0) > 0)[0]
        x1, x2 = max(cols[0] - MARGIN, 0), cols[-1] + MARGIN + 1
        y1, y2 = max(a - MARGIN_Y, 0), b + MARGIN_Y
        lines.append(gray[y1:y2, x1:x2])
    return lines, angle, skipped


def main():
    os.makedirs("images", exist_ok=True)
    make_excel = not os.path.exists("data/real_lines.xlsx")
    if not make_excel:
        print("data/real_lines.xlsx already exists - only the line images are made again "
              "(your typed text is safe).")

    wb = Workbook()
    ws = wb.active
    ws.title = "lines"
    ws.append(["file", "lang", "image", "text"])
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["C"].width = 110
    ws.column_dimensions["D"].width = 100

    row = 2
    for lang, path in PHOTOS.items():
        lines, angle, skipped = cut_page(path)
        print(f"{lang}: {len(lines)} lines, page rotated {angle} degrees, "
              f"{skipped} blocks of bent lines skipped")
        for i, line in enumerate(lines, start=1):
            name = f"real_{lang}_{i:02d}.png"
            cv2.imwrite(os.path.join("images", name), line)
            ws.cell(row=row, column=1, value=name)
            ws.cell(row=row, column=2, value=lang)
            cell = ws.cell(row=row, column=4)
            cell.font = Font(size=14)
            cell.alignment = Alignment(vertical="center", readingOrder=2 if lang == "fa" else 1)
            pic = XLImage(os.path.join("images", name))
            scale = min(1.0, 750 / pic.width)
            pic.width, pic.height = pic.width * scale, pic.height * scale
            ws.add_image(pic, f"C{row}")
            ws.row_dimensions[row].height = max(pic.height * 0.75 + 8, 22)
            row += 1

    if make_excel:
        wb.save("data/real_lines.xlsx")
        print("saved data/real_lines.xlsx - type the correct text of each line in column D")


if __name__ == "__main__":
    main()
