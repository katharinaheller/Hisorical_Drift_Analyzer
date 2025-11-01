import os
from PyPDF2 import PdfFileReader
from pathlib import Path

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def get_file_size(file_path: Path) -> int:
    """Returns the file size in bytes."""
    return os.path.getsize(file_path)

def get_pdf_page_count(file_path: Path) -> int:
    """Returns the number of pages in a PDF."""
    with open(file_path, "rb") as file:
        reader = PdfFileReader(file)
        return reader.getNumPages()
