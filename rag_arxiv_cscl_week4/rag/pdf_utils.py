import fitz #PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """ Extract text from a PDF file """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        pages.append(text)
    return "\n".join(pages)

