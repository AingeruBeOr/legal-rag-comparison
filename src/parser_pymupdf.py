from langchain_community.document_loaders import PyMuPDFLoader

class PDFParser:
    def __init__(self):
        pass

    def parse(self, file_path):
        """Parse the PDF and return a list of documents (one per page)."""
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents