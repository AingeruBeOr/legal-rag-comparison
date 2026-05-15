"""
Documentación: https://ai.google.dev/gemini-api/docs
"""

from google import genai

class GeminiInferencer:
    def __init__(self):
        self.client = genai.Client()
        pass

    def infer(self, prompt) -> str:
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite", contents=prompt
        )
        return response.text