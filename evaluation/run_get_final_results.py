import json
import math
import os

# Ruta al archivo JSON de evaluación
JSON_PATH = os.path.abspath("../output/evaluation_results/eval_20260611_220459.json")
JSON_PATH = os.path.abspath("../output/checkpoints/eval_20260612_153807.json")


def calcular_media_y_desviacion(valores):
    """Calcula la media aritmética y la desviación típica de una lista de números."""
    if not valores:
        return 0.0, 0.0
    
    n = len(valores)
    media = sum(valores) / n
    
    # Desviación típica poblacional
    varianza = sum((x - media) ** 2 for x in valores) / n
    desviacion = math.sqrt(varianza)
    
    return media, desviacion

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: No se encuentra el archivo en {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        print("No se encontraron 'qa_pairs' en el archivo.")
        return

    # Inicializar diccionarios para acumular los valores numéricos
    metricas = {
        "time_taken_seconds": [],
        "input_tokens": [],
        "output_tokens": [],
        "context_relevance": [],
        "groundedness": [],
        "answer_relevance": [],
        "hit_rate": [],
        "mrr": [],
        "bleu": [],
        "rouge_l": [],
        "bertscore_f1": []
    }

    evaluated_count = 0

    # Extraer los datos de cada par de pregunta/respuesta
    for qa in qa_pairs:
        # 1. Métricas generales del paso de generación
        for key in ["time_taken_seconds", "input_tokens", "output_tokens"]:
            if qa.get(key) is not None:
                metricas[key].append(float(qa[key]))

        # 2. Métricas del LLM Judge y NLP dentro de 'evaluation'
        evaluation = qa.get("evaluation")
        if evaluation:
            for key in evaluation.keys():
                if key in metricas:
                    val = evaluation[key]
                    # Si la métrica es un diccionario (como context_relevance que tiene 'score' y 'reasoning')
                    if isinstance(val, dict):
                        score = val.get("score")
                        if score is not None:
                            metricas[key].append(float(score))
                    # Si es un valor numérico directo (como bleu, mrr, hit_rate...)
                    elif isinstance(val, (int, float)):
                        metricas[key].append(float(val))
            evaluated_count += 1

    # Imprimir los resultados formateados en una tabla limpia
    print("\n" + "="*65)
    print(f"{'MÉTRICA':<25} | {'MEDIA':<15} | {'DESV. TÍPICA (σ)':<15}")
    print("="*65)

    for metrica, valores in metricas.items():
        if valores:
            media, desviacion = calcular_media_y_desviacion(valores)
            # Formatear la salida (los tokens como enteros si se prefiere, aquí con decimales para precisión)
            print(f"{metrica:<25} | {media:<15.4f} | {desviacion:<15.4f}")
        else:
            print(f"{metrica:<25} | {'Sin datos':<15} | {'Sin datos':<15}")
            
    print("="*65)
    print(f"Total de registros evaluados: {evaluated_count}/{len(qa_pairs)}\n")

if __name__ == "__main__":
    main()