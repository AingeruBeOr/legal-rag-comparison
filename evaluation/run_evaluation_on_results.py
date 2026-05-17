import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from time import sleep
from datetime import datetime
from dotenv import load_dotenv
load_dotenv("../.env")
from tqdm import tqdm
from evaluation.RAGEvaluator import RAGEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class Config:
    # Ruta del archivo generado por do_rag_test_set.py
    # NOTA: Cambia esto por el nombre del archivo real que quieras evaluar
    RESULTS_PATH = os.path.abspath("../output/generation_results/20260517_125910.json")
    
    LOAD_FROM_CHECKPOINT = False
    if LOAD_FROM_CHECKPOINT:
        # introduce manualmente el checkpoint a cargar
        CHECKPOINT_PATH = os.path.abspath(f"../output/checkpoints/eval_XXX.json")
    else:
        # crea un nuevo checkpoint para esta ejecución
        CHECKPOINT_PATH = os.path.abspath(f"../output/checkpoints/eval_{TIMESTAMP}.json")      

    SLEEP_TIME = 2 # para evitar rate limits del LLM Judge


def main():
    if not os.path.exists(Config.RESULTS_PATH):
        print(f"Error: El archivo de resultados {Config.RESULTS_PATH} no existe. Por favor, actualiza Config.RESULTS_PATH.")
        return

    evaluator = RAGEvaluator()

    with open(Config.RESULTS_PATH, "r") as f:
        data = json.load(f)

    # Imprimir configuración
    print(f"Configuración de la evaluación:")
    for key, value in Config.__dict__.items():
        if not key.startswith("__") and not callable(value):
            print(f"{key}: {value}")
    input("Pulsa Enter para iniciar la evaluación...")

    results = data.copy()

    # Cargar checkpoint si existe y está habilitado
    if Config.LOAD_FROM_CHECKPOINT and os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Cargando checkpoint desde {Config.CHECKPOINT_PATH}...")
        with open(Config.CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data

    # Diccionario para acceder rápido a los pares ya evaluados con éxito
    processed_queries = {}
    for qa in results.get("qa_pairs", []):
        if qa.get("evaluation") is not None:
            query_key = qa.get("question")
            processed_queries[query_key] = qa

    # Asegurar que el directorio de checkpoints existe
    os.makedirs(os.path.dirname(Config.CHECKPOINT_PATH), exist_ok=True)

    errors_count = 0
    qa_pairs = results.get("qa_pairs", [])
    
    # Comprobar que no hay respuestas generadas vacías (None)
    if any(qa.get("generated_answer") is None for qa in qa_pairs):
        print("Error: Se encontraron preguntas sin 'generated_answer' (es None).")
        print("Asegúrate de que el script de generación se haya completado correctamente con todas las respuestas antes de evaluar.")
        return

    for i, qa in enumerate(tqdm(qa_pairs, desc="Evaluando preguntas")):
        query = qa.get("question")
        
        # Saltar si ya tenemos una evaluación válida en el checkpoint
        if Config.LOAD_FROM_CHECKPOINT and query in processed_queries:
            continue

        # Si no hubo respuesta generada válidamente en el paso anterior, no tiene sentido evaluar
        if not qa.get("generated_answer"):
            qa["evaluation"] = None
            continue

        try:
            eval_metrics = evaluator.evaluate(qa)
            # Guardamos las métricas dentro de una nueva key 'evaluation'
            qa["evaluation"] = eval_metrics
        except Exception as e:
            print(f"\nError al evaluar la pregunta '{query}': {e}")
            qa["evaluation"] = None
            errors_count += 1

        # Guardar checkpoint después de cada petición 
        with open(Config.CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        break # to test
        sleep(Config.SLEEP_TIME)

    print(f"\nEvaluación completada con {errors_count} errores.")

    os.makedirs("../output/evaluation_results", exist_ok=True)
    final_results_path = os.path.abspath(f"../output/evaluation_results/eval_{TIMESTAMP}.json")
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Guadados los resultados finales en: {final_results_path}")

if __name__ == "__main__":
    main()