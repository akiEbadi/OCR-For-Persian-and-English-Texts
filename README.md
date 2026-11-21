# Comparing Tesseract and PaddleOCR for Persian and English Text Recognition

This project compares two OCR tools on reading single lines of text. It checks how well each tool reads Persian and English, which settings work for each language, and how the inference parameters change the result.

This README starts from the moment you have unzipped the project folder.

---

## 1. Folder structure

```
ocr-project/
├── data/
│   ├── sentences_fa.txt        30 Persian sentences
│   ├── sentences_en.txt        30 English sentences
│   ├── real_photos/            two page photos: fa_page.png and en_page.jpg
│   └── real_lines.xlsx         the correct text of the photo lines (column D)
├── models/tessdata_best/       fas.traineddata and eng.traineddata
├── images/                     generated images (created automatically)
├── results/                    results (created automatically)
├── check_install.py            install test
├── make_images.py              makes images from the sentences
├── cut_lines.py                cuts lines out of the page photos
├── run_ocr.py                  runs Tesseract and PaddleOCR
├── evaluate.py                 computes CER and WER
├── test_params.py              tests the inference parameters
└── requirements.txt            Python libraries
```

---

## 2. Requirements

- Windows with WSL and Ubuntu
- Internet access for installing libraries and for the PaddleOCR models (first run only)
- Python 3.12 (PaddlePaddle does not install on Python 3.14)

---

## 3. Setup

Run all commands in this section inside Ubuntu.

### 3.1 Install system packages

```bash
sudo apt update
sudo apt install -y tesseract-ocr fonts-vazirmatn fonts-farsiweb fonts-liberation fonts-dejavu libraqm0 python3-venv
```

### 3.2 Go to the project folder

```bash
cd ~/ocr-project
```

### 3.3 Create a Python 3.12 environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 3.4 Install the libraries

```bash
pip install -r requirements.txt
```

The packages are `paddlepaddle` and `paddleocr` (PaddleOCR), `pytesseract` (Python link to Tesseract), `pillow` and `opencv-python-headless` (image handling), `numpy` and `pandas` (calculations and tables), `jiwer` (CER and WER), and `openpyxl` (Excel files).

### 3.5 Add the Tesseract models

```bash
mkdir -p models/tessdata_best
cp /path/to/fas.traineddata /path/to/eng.traineddata models/tessdata_best/
```

Download both files from `github.com/tesseract-ocr/tessdata_best` first, then copy them into the `models/tessdata_best` folder.

The PaddleOCR models do not need to be added by hand. They download automatically the first time you run the project.

### 3.6 Put in your corrected photo text

If you have your own `real_lines.xlsx`, copy it into the `data` folder:

```bash
cp /mnt/c/Users/<your-windows-user-name>/Downloads/real_lines.xlsx data/real_lines.xlsx
```

---

## 4. Running the programs in order

Run these commands from the `~/ocr-project` folder with the environment activated.

1. `python check_install.py`: tests that both tools work. Output: `ALL OK`
2. `python make_images.py`: makes images from the sentences. Output: `images made: 240`
3. `python cut_lines.py`: cuts the lines out of the page photos. Output: line counts for Persian and English
4. `python run_ocr.py`: reads all images with both tools. Output: `saved results/predictions.csv`
5. `python evaluate.py`: computes CER and WER. Output: a results table
6. `python test_params.py`: tests the parameters. Output: a parameter table and `saved results/params.csv`

---

## 5. What CER and WER mean

- **CER (character error rate):** the percentage of characters that were read wrongly. Lower is better.
- **WER (word error rate):** the percentage of words that were read wrongly. Lower is better.

---

## 6. Common problems

| Message or problem | Cause | Fix |
| --- | --- | --- |
| `No matching distribution found for paddlepaddle` | Python is 3.14 | Create the environment again with `uv venv --python 3.12 .venv` |
| `externally-managed-environment` | The system Python is locked | Activate `.venv` first, then install with `pip install -r requirements.txt` |
| `tesseract: command not found` | Tesseract is not installed | Run step 3.1 again |
| `Failed to load traineddata` or language `fas` not found | The model files are missing from `models/tessdata_best` | Do step 3.5 |
| PaddleOCR model download stops | The internet connection is unstable | Run the command again. The models are saved in `~/.paddlex` |
| `UserWarning: No ccache found` | A warning only | Ignore it |
| `ReduceMeanCheckIfOneDNNSupport` | An information message | Ignore it |
