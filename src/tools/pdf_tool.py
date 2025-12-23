from langchain.tools import tool
from pypdf import PdfReader
import os

@tool
def read_pdf(file_path: str) -> str:
    """
    Reads the text content from a PDF file.
    
    Args:
        file_path (str): The absolute path to the PDF file.
        
    Returns:
        str: The text content of the PDF, or an error message if reading fails.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
            
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def get_pdf_tool():
    return read_pdf
