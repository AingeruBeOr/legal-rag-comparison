from langchain_community.document_loaders import UnstructuredPDFLoader

class PDFParser:
    def __init__(self):
        pass

    def parse(self, file_path):
        """Parse the PDF and return a list of documents (one per page)."""
        loader = UnstructuredPDFLoader(file_path, 
                                       strategy="fast",  # Usa "fast" para PDFs digitales, evita el OCR innecesario
                                       )
        documents = loader.load()
        return documents