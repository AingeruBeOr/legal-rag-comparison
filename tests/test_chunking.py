import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chunking import PDFChunker

documents = [{"page_content": "This is a test document. " * 100, "metadata": {"page": 1}}]
chunker = PDFChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(documents)
print("Número de chunks:", len(chunks))
print("\n--- Page content of the first chunk ---")
print(chunks[0].page_content[:500])
print("\n\n--- Metadata of the first chunk ---")
print(chunks[0].metadata)