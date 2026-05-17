"""
Execute: `python do_rag_test_set.py` 
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from time import time, sleep
from datetime import datetime
from dotenv import load_dotenv
load_dotenv("../.env")
from tqdm import tqdm
from src.online_pipeline import OnlinePipeline

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class Config:
    TEST_SET_PATH = os.path.abspath("../data/test_set.json")
    LOAD_FROM_CHECKPOINT = False
    if LOAD_FROM_CHECKPOINT:
        # introduce manually the checkpoint to run from
        CHECKPOINT_PATH = os.path.abspath(f"../output/checkpoints/rag_XXX.json")
    else:
        # create a new checkpoint path
        CHECKPOINT_PATH = os.path.abspath(f"../output/checkpoints/rag_{TIMESTAMP}.json")      

    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    TOP_K = 3
    COLLECTION_NAME = "documents"
    LLM_MODEL_NAME = "gemini-3.1-flash-lite"
    SLEEP_TIME = 3 # to avoid hitting rate limits

test_set_path = os.path.abspath(Config.TEST_SET_PATH)
with open(test_set_path, "r") as f:
    test_set = json.load(f)

# print all the config values
print(f"RAG Pipeline Config:")
for key, value in Config.__dict__.items():
    if not key.startswith("__") and not callable(value):
        print(f"{key}: {value}")
input("Press Enter to run the RAG test...")

rag_pipeline = OnlinePipeline(
    embedding_model_name=Config.EMBEDDING_MODEL_NAME,
    top_k=Config.TOP_K, 
    collection_name=Config.COLLECTION_NAME,
    llm_model_name=Config.LLM_MODEL_NAME,
)

results = {
    "config": {
        "embedding_model_name": Config.EMBEDDING_MODEL_NAME,
        "top_k": Config.TOP_K,
        "collection_name": Config.COLLECTION_NAME,
        "llm_model_name": Config.LLM_MODEL_NAME,
    },
    "qa_pairs": []
}

# Cargar checkpoint si existe y está habilitado
if Config.LOAD_FROM_CHECKPOINT and os.path.exists(Config.CHECKPOINT_PATH):
    print(f"Cargando checkpoint desde {Config.CHECKPOINT_PATH}...")
    with open(Config.CHECKPOINT_PATH, "r") as f:
        checkpoint_data = json.load(f)
        # Actualizamos resultados con el checkpoint
        results = checkpoint_data

# Crear un diccionario para acceder rápido a los pares ya procesados con éxito
processed_queries = {}
for qa in results["qa_pairs"]:
    if qa.get("generated_answer") is not None:
        query_key = qa.get("question")
        processed_queries[query_key] = qa

# Asegurar que el directorio de checkpoints existe
os.makedirs(os.path.dirname(Config.CHECKPOINT_PATH), exist_ok=True)

errors_count = 0
for i, qa in enumerate(tqdm(test_set, desc="Procesando preguntas")):
    query = qa.get("question")
    
    # Si ya tenemos una respuesta válida en el checkpoint, la saltamos
    if Config.LOAD_FROM_CHECKPOINT and query in processed_queries:
        continue

    try:
        generated_answer, retrieved_contexts = rag_pipeline.do_rag(query=query)
    except Exception as e:
        print(f"\nError al procesar la pregunta '{query}': {e}")
        generated_answer = None
        errors_count += 1

    qa_result = {**qa, "generated_answer": generated_answer,
                 "retrieved_context": [{"score": rc.score, "payload": rc.payload} 
                                       for rc in retrieved_contexts] if generated_answer else None}
    
    # Si estaba en results (por un intento fallido previo), reemplazarlo. Si no, añadirlo.
    existing_index = next((idx for idx, item in enumerate(results["qa_pairs"]) if (item.get("question")) == query), None)
    if existing_index is not None:
        # if loading from checkpoint
        results["qa_pairs"][existing_index] = qa_result
    else:
        results["qa_pairs"].append(qa_result)

    # Guardar checkpoint después de cada petición 
    with open(Config.CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    break # to test
    sleep(Config.SLEEP_TIME)

print(f"\nRAG test completado con {errors_count} errores.")

os.makedirs("../output/generation_results", exist_ok=True)
results_path = os.path.abspath(f"../output/generation_results/{TIMESTAMP}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Resultados guardados en {results_path}")