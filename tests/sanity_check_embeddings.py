"""
Sanity-check para diagnosticar problemas de retrieval con MeanPoolingEmbeddings.

Qué hace:
    1. Calcula el embedding de una query.
    2. Calcula el embedding de un conjunto de chunks que TÚ marcas a mano
       como "relevantes" o "irrelevantes" para esa query.
    3. Muestra el coseno de cada chunk contra la query, ordenado de mayor
       a menor, indicando si era relevante o no.
    4. Da un diagnóstico automático:
         - ¿Los relevantes están arriba del ranking?
         - ¿Hay separación real entre relevantes/irrelevantes, o todo
           está apelmazado en un rango estrecho (síntoma de anisotropía
           típico de modelos MLM sin fine-tuning contrastivo)?

Cómo usarlo:
    1. Edita la sección "EDITA AQUÍ ABAJO" con tu query y tus chunks.
    2. Cambia MODEL_NAME si quieres probar otro modelo (p.ej. MrBERT-es).
    3. Ejecuta: python sanity_check_embeddings.py

Importante: usa la MISMA configuración (model_name, max_length, normalize)
que usas en tu pipeline de ingestión real. Si difieren, este test no es
representativo de lo que está pasando en tu RAG.
"""

from typing import List, Tuple
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_mean_pooling import MeanPoolingEmbeddings


# =============================================================================
# EDITA AQUÍ ABAJO
# =============================================================================

MODEL_NAME = "BSC-LT/MrBERT-legal"  # cambia a "BSC-LT/MrBERT-es" para comparar
MAX_LENGTH = 1024
NORMALIZE = True

QUERY = "¿Qué dice la ley sobre indemnización por daños y perjuicios?"

# Chunks que SABES que son relevantes para la query de arriba
RELEVANT_CHUNKS = [
    "El artículo 1902 del Código Civil establece que el que por acción u "
    "omisión causa daño a otro, interviniendo culpa o negligencia, está "
    "obligado a reparar el daño causado.",
    "La indemnización por daños y perjuicios debe cubrir tanto el daño "
    "emergente como el lucro cesante derivado del incumplimiento.",
]

# Chunks que SABES que NO tienen relación con la query
IRRELEVANT_CHUNKS = [
    "El Boletín Oficial del Estado publica hoy el calendario laboral "
    "para el próximo ejercicio fiscal.",
    "La receta de la tortilla de patatas lleva huevos, patatas y cebolla, "
    "según la tradición culinaria española.",
    "El equipo de fútbol ganó el partido por dos goles a cero en el "
    "estadio local.",
]

# =============================================================================
# FIN DE LA SECCIÓN EDITABLE
# =============================================================================


def cosine_sim(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def main():
    print(f"Modelo: {MODEL_NAME}")
    print(f"max_length={MAX_LENGTH}  normalize={NORMALIZE}")
    print("=" * 80)

    embedder = MeanPoolingEmbeddings(
        model_name=MODEL_NAME,
        show_progress=False,
        max_length=MAX_LENGTH,
        normalize=NORMALIZE,
    )

    query_emb = embedder.embed_query(QUERY)

    labeled_chunks: List[Tuple[str, str]] = (
        [(c, "RELEVANTE") for c in RELEVANT_CHUNKS]
        + [(c, "irrelevante") for c in IRRELEVANT_CHUNKS]
    )

    texts = [c for c, _ in labeled_chunks]
    chunk_embs = embedder.embed_documents(texts)

    results = []
    for (text, label), emb in zip(labeled_chunks, chunk_embs):
        sim = cosine_sim(query_emb, emb)
        results.append((sim, label, text))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\nQuery: {QUERY}\n")
    print(f"{'COSENO':>8}  {'ETIQUETA':<12}  TEXTO")
    print("-" * 80)
    for sim, label, text in results:
        snippet = text if len(text) <= 70 else text[:67] + "..."
        print(f"{sim:8.4f}  {label:<12}  {snippet}")

    # --- Diagnóstico automático ---
    sims_relevant = [s for s, l, _ in results if l == "RELEVANTE"]
    sims_irrelevant = [s for s, l, _ in results if l == "irrelevante"]

    mean_rel = np.mean(sims_relevant)
    mean_irrel = np.mean(sims_irrelevant)
    gap = mean_rel - mean_irrel

    all_sims = [s for s, _, _ in results]
    spread = max(all_sims) - min(all_sims)

    # ¿Todos los relevantes están por encima de todos los irrelevantes?
    perfect_separation = min(sims_relevant) > max(sims_irrelevant)

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO")
    print("=" * 80)
    print(f"Coseno medio (relevantes):   {mean_rel:.4f}")
    print(f"Coseno medio (irrelevantes): {mean_irrel:.4f}")
    print(f"Diferencia (gap):            {gap:.4f}")
    print(f"Rango total de cosenos:      {spread:.4f}  "
          f"(min={min(all_sims):.4f}, max={max(all_sims):.4f})")
    print()

    if perfect_separation and gap > 0.05:
        print("✅ Separación CLARA: todos los relevantes superan a todos los "
              "irrelevantes, con un margen razonable.")
        print("   El modelo SÍ está discriminando semánticamente para este "
              "ejemplo. Si en producción los resultados siguen sin tener "
              "sentido, sospecha de un desajuste entre el embedder usado "
              "para indexar y el usado para consultar, o de un problema en "
              "el vectorstore (índice corrupto, distancia mal configurada, "
              "metadata desalineada con los vectores, etc.).")
    elif gap > 0.03:
        print("⚠️  Separación DÉBIL: en promedio los relevantes puntúan algo "
              "más alto, pero hay solapamiento (no todos los relevantes "
              "superan a todos los irrelevantes).")
        print("   Esto es compatible con anisotropía parcial: el modelo "
              "captura algo de señal semántica pero con mucho ruido. Para "
              "retrieval en producción esto suele traducirse en chunks "
              "recuperados que 'casi' tienen sentido o que mezclan ruido.")
    else:
        print("❌ SIN separación significativa: relevantes e irrelevantes "
              "obtienen cosenos prácticamente iguales.")
        print("   Esto es el patrón clásico de anisotropía en embeddings "
              "de un modelo MLM (entrenado con masked-language-modeling, "
              "sin fine-tuning contrastivo tipo sentence-transformers). El "
              "mean pooling sobre estos modelos tiende a producir vectores "
              "muy parecidos entre sí sin importar el contenido, dominados "
              "por la norma/frecuencia de tokens en lugar del significado.")
        print()
        print("   Recomendación: prueba este mismo script cambiando "
              "MODEL_NAME a 'BSC-LT/MrBERT-es' (mejor puntuación de "
              "retrieval en español según el model card) o a un modelo "
              "explícitamente entrenado para embeddings/retrieval.")

    if spread < 0.05:
        print()
        print("ℹ️  Nota adicional: el rango total de cosenos es muy estrecho "
              f"({spread:.4f}). Esto por sí solo es un indicador fuerte de "
              "anisotropía: todos los vectores apuntan casi en la misma "
              "dirección en el espacio de embeddings, así que cualquier "
              "ranking de similitud será casi aleatorio en la práctica, "
              "aunque alguno de los valores parezca técnicamente más alto.")


if __name__ == "__main__":
    main()
