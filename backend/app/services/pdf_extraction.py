try:
    import pymupdf as fitz
except ImportError:
    import fitz

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a given PDF file using PyMuPDF.
    
    Args:
        file_path (str): The absolute or relative path to the PDF file.
        
    Returns:
        str: The extracted text.
    """
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    return text
