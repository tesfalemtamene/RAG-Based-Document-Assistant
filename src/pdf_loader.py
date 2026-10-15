from pypdf import PdfReader

def load_pdf(file_path):
    """
    Extract text from a PDF file
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text