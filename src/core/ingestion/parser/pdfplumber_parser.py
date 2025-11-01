import pytesseract
import fitz  # PyMuPDF
import pdfplumber  # Import pdfplumber for better multi-column text extraction
from PIL import Image, ImageEnhance, ImageFilter
import os
import re
from typing import Dict, Any, List
from pathlib import Path
from src.core.ingestion.parser.interfaces.i_pdf_parser import IPdfParser

class PdfPlumberParser(IPdfParser):
    """Robust PDF parser using pdfplumber to handle multi-column text extraction and OCR fallback."""

    def __init__(self, exclude_toc: bool = True, max_pages: int | None = None, use_ocr: bool = True):
        """
        Initialize the parser with options for excluding the table of contents and limiting the number of pages.
        
        :param exclude_toc: Flag to exclude table of contents pages from extraction.
        :param max_pages: Maximum number of pages to extract text from.
        :param use_ocr: Flag to use OCR if no text is found.
        """
        self.exclude_toc = exclude_toc
        self.max_pages = max_pages
        self.use_ocr = use_ocr  # Flag to use OCR if no text is found

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract clean body text (including abstract) and minimal metadata from a PDF file.

        :param pdf_path: Path to the PDF file to extract text from.
        :return: A dictionary with 'text' (extracted body text) and 'metadata' (file metadata).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        text_blocks: List[str] = []  # To hold the extracted text from each page
        toc_titles = self._extract_toc_titles(pdf_path) if self.exclude_toc else []  # Extract TOC titles to filter out

        # Extract body text from each page using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                if self.max_pages and page_index >= self.max_pages:
                    break

                # Extract text from the page considering multi-column layout
                page_body = self._extract_text_from_page(page)
                if page_body:
                    text_blocks.append(page_body)

        clean_text = "\n".join(t for t in text_blocks if t)
        clean_text = self._remove_residual_metadata(clean_text)

        metadata = {
            "source_file": str(pdf_path.name),
            "page_count": len(pdf.pages),
        }

        return {"text": clean_text.strip(), "metadata": metadata}

    def _extract_text_from_page(self, page) -> str:
        """
        Extract text from a page considering the multi-column layout using pdfplumber.
        
        :param page: The pdfplumber page object.
        :return: Extracted text from the page.
        """
        # Split the page into columns using pdfplumber's layout extraction
        width = page.width
        left_column = page.within_bbox((0, 0, width / 3, page.height))  # Left column
        middle_column = page.within_bbox((width / 3, 0, 2 * width / 3, page.height))  # Middle column
        right_column = page.within_bbox((2 * width / 3, 0, width, page.height))  # Right column

        # Extract text from each column
        left_text = left_column.extract_text()
        middle_text = middle_column.extract_text()
        right_text = right_column.extract_text()

        # Combine the text from all columns
        combined_text = f"{left_text}\n{middle_text}\n{right_text}"

        return combined_text

    def _extract_toc_titles(self, pdf_path: Path) -> List[str]:
        """
        Extract table of contents (TOC) titles from the PDF document to filter them out from the main text.
        
        :param pdf_path: The path to the PDF file.
        :return: A list of TOC titles.
        """
        toc_titles = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try to get TOC titles from the page (if any)
                toc = page.extract_text()
                if toc and re.search(r"(?i)\btable\s+of\s+contents\b", toc):
                    toc_titles.append(toc)
        return toc_titles

    def _remove_residual_metadata(self, text: str) -> str:
        """
        Remove residual header/footer and metadata-like fragments, excluding abstract.
        
        :param text: The extracted text to clean up.
        :return: Cleaned text with metadata removed.
        """
        patterns = [
            r"(?im)^table\s+of\s+contents.*$",
            r"(?im)^inhaltsverzeichnis.*$",
            r"(?im)^\s*(title|author|doi|keywords|arxiv|university|faculty|institute|version).*?$",
            r"(?im)\bpage\s+\d+\b",  # Remove page numbers
            r"(?m)^\s*\d+\s*$",  # Remove single digit page numbers like "1", "2", "3"
            r"(?i)\bhttps?://\S+",  # Remove URLs
            r"(?i)\b\w+@\w+\.\w+",  # Remove email addresses
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.DOTALL)

        text = re.sub(r"\n{2,}", "\n\n", text)  # Merge consecutive line breaks
        return text.strip()

    def _apply_ocr_to_page(self, page) -> List[str]:
        """
        Apply OCR to a page if the page does not contain text. Uses Tesseract to extract text from scanned pages.
        
        :param page: The page to apply OCR on.
        :return: A list with OCR extracted text.
        """
        # Convert the page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Preprocess the image (optional, if necessary)
        img = self._preprocess_image_for_ocr(img)

        # Apply OCR
        ocr_text = pytesseract.image_to_string(img)
        return [(0, 0, pix.width, pix.height, ocr_text)]  # Return OCR'd text as a block

    def _preprocess_image_for_ocr(self, img: Image) -> Image:
        """
        Preprocess the image to improve OCR accuracy by enhancing contrast and sharpness.
        
        :param img: The image to preprocess.
        :return: The preprocessed image ready for OCR.
        """
        # Sharpen the image
        img = img.filter(ImageFilter.SHARPEN)
        
        # Increase the contrast of the image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)  # Enhance the contrast by a factor of 2
        
        return img
