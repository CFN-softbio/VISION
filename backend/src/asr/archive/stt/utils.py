
import fitz  # PyMuPDF
import os

def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text
