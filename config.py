import os

tesseract_path = os.environ.get('TESSDATA_PREFIX', r'C:\Users\katha\AppData\Local\Programs\Tesseract-OCR')

import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_path, 'tesseract.exe')

try:
    print("Tesseract-Version:", pytesseract.get_tesseract_version())
except Exception as e:
    print("Fehler beim Laden von Tesseract:", e)
