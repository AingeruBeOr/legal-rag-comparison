from docling.document_converter import DocumentConverter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

class PDFParser:
    def __init__(self):
        self.converter = DocumentConverter()
        pass

    def parse(self, file_path):
        """Parse the PDF and return a list of documents (one per page)."""
        loader = DoclingLoader(
            file_path=file_path,
            converter=self.converter,
            export_type=ExportType.MARKDOWN,  # o DOC_CHUNKS con tokenizer correcto
        )
        documents = loader.load()
        return documents