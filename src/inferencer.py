"""
Documentación: https://ai.google.dev/gemini-api/docs
"""

from google import genai

class GeminiInferencer:
    def __init__(self, model_id="gemini-3.1-flash-lite"):
        self.client = genai.Client()
        self.model_id = model_id

    def infer(self, prompt) -> str:
        response = self.client.models.generate_content(
            model=self.model_id, contents=prompt
        )
        return response.text