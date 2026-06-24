from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

class PDFChunker:
    """
    def __init__(self, threshold_type: str = "percentile", threshold_amount: int = 50):
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount,
        )

    def chunk(self, pdf_text):

        chunks = self.chunker.split_documents(pdf_text)

        return chunks
    """
    
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            separator="",           #  "" for exact character count, " " for word count
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def chunk(self, documents):
        documents = self.text_splitter.split_documents(documents)
        return documents
        