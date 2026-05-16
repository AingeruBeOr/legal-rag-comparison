from langchain_text_splitters import CharacterTextSplitter

class PDFChunker:
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
        