import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_gemini import GeminiEmbedder
from dotenv import load_dotenv
load_dotenv("../.env")

embedder = GeminiEmbedder()

response = embedder.embed("What is the capital of France?")
print(response)