import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("../.env")

class OpenRouterInferencer:
    def __init__(self, api_key: str = None, model_id="openai/gpt-5.4-mini", max_tokens=500):
        """
        Inicializa el inferenciador utilizando la API de OpenRouter.
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key if api_key else os.getenv("OPENROUTER_API_KEY"),
        )
        self.model_id = model_id
        self.max_tokens = max_tokens

    def infer(self, prompt: str) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            # Headers recomendados por OpenRouter para identificar tu proyecto
            extra_headers={
                "X-Title": "TFM Legal RAG Evaluation",
            }
        )
        
        # Extraer el texto de la respuesta (formato OpenAI)
        texto_respuesta = response.choices[0].message.content
        
        # Estructurar los tokens consumidos de forma idéntica a tu clase anterior
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        
        return texto_respuesta, tokens