import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inferencer_openrouter import OpenRouterInferencer
from dotenv import load_dotenv
load_dotenv("../.env")

# Instancias con tu clave y el modelo seleccionado
inferenciador = OpenRouterInferencer(
    model_id="openai/gpt-5.4-mini",
    max_tokens=400 # Ajustado para evitar el error de saldo que vimos antes
)

# Ejecutas la inferencia sobre tus documentos legales o prompts
texto, metadata_tokens = inferenciador.infer("Analiza el siguiente artículo...")

print(texto)
print(metadata_tokens) # {'input_tokens': XX, 'output_tokens': XX}