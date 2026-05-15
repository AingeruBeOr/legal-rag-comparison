"""
Reference: https://ai.google.dev/gemini-api/docs/embeddings
"""

from google import genai
from google.genai import types

class GeminiEmbedder:
    def __init__(self):
        self.client = genai.Client()
        self.vector_size = 768 # can be adjusted based on the model's output dimensionality (check documentation)

    def embed(self, text) -> list:
        response = self.client.models.embed_content(
            model="gemini-embedding-2", contents=text, 
            config=types.EmbedContentConfig(output_dimensionality=self.vector_size)
        )
        return response.embeddings