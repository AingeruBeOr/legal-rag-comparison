import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
from src.embedding import HuggingFaceEmbeddingsLC

embedder = HuggingFaceEmbeddingsLC()

text = "Hola, ¿cómo estás?"
start_time = time()
embedding = embedder.embed(text)
total_time = time() - start_time
print(f"Embedding generado en {total_time:.2f} segundos.")
print(f"Embedding for '{text}':\n{embedding}")