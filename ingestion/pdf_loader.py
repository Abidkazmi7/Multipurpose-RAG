from pypdf import PdfReader
from langchain_core.documents import Document

# Extracts raw text content from PDF 
def extract_text(pdf_path): 
    reader = PdfReader(pdf_path) 
    text = [] 

    for page in reader.pages: 
        extracted = page.extract_text() 

        if extracted: 
            text.append(extracted) 

    return "\n".join(text) 

# Obtains LangChain document object with it's associated metadata 
def load_pdf(pdf_path):
    text = extract_text(pdf_path) 
    
    return [Document(page_content = text, metadata = {"source" : pdf_path})]