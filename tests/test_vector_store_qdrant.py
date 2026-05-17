import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vector_store_qdrant import QdrantClient

qdrant_client = QdrantClient("http://localhost:6333")
info = qdrant_client.get_collection_info("documents")

print(info.config.metadata['embedding_model'])