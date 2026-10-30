import fitz


def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    return "\n\n".join(pages).strip()
