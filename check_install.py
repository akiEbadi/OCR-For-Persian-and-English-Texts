# Step 0: check that Tesseract and PaddleOCR are installed and work.
# Writes one Persian and one English test line, reads them with both tools
# and prints PASS if the result is close to the original text.

import os
import subprocess

import jiwer
import paddle
import paddleocr

from evaluate import normalize
from make_images import FONTS, draw_line
from run_ocr import TESS_DATA, load_paddle_models, run_paddle, run_tesseract

TEST_LINE = {
    "fa": "این یک آزمایش ساده برای بررسی نصب نرم‌افزار است.",
    "en": "This is a simple test to check the installation.",
}


def main():
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout
    langs = subprocess.run(["tesseract", "--list-langs", "--tessdata-dir", TESS_DATA],
                           capture_output=True, text=True).stdout.split()[-2:]
    print("Tesseract :", version.splitlines()[0], "| languages:", langs)
    print("PaddleOCR :", paddleocr.__version__, "| PaddlePaddle:", paddle.__version__)

    os.makedirs("results/check", exist_ok=True)
    paddle_models = load_paddle_models()
    all_ok = True
    for lang, text in TEST_LINE.items():
        font_path = list(FONTS[lang].values())[0]
        path = f"results/check/test_{lang}.png"
        draw_line(text, font_path).save(path)

        for tool, output in [("tesseract", run_tesseract(path, lang)),
                             ("paddle", run_paddle(paddle_models[lang], path))]:
            cer = jiwer.cer(normalize(text), normalize(output)) * 100
            ok = cer < 10
            all_ok = all_ok and ok
            print(f"{'PASS' if ok else 'FAIL'}  {lang}  {tool:9s}  CER {cer:5.1f}%  ->  {output}")

    print("\nALL OK" if all_ok else "\nSOMETHING IS WRONG")


if __name__ == "__main__":
    main()
