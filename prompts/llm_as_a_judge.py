prompt = """
Actúa como un tribunal de evaluación científico y neutral para sistemas de Recuperación de Información (RAG) en el dominio legal español. 

Analiza los siguientes tres elementos:
- Pregunta del Usuario: {query}
- Contexto Recuperado del BOE: {context}
- Respuesta Generada por el Sistema: {answer}

Tu tarea es evaluar tres métricas independientes. Para evitar sesgos, procesa cada métrica paso a paso siguiendo estas directrices:

1. RELEVANCIA DEL CONTEXTO (Context Relevance):
   - Evalúa si el 'Contexto Recuperado' contiene la información necesaria para responder a la 'Pregunta', ignorando la 'Respuesta'.
   - Justifica tu análisis brevemente en una frase.
   - Asigna una puntuación de 0.0 (totalmente irrelevante) a 1.0 (contiene toda la información necesaria).

2. FIDELIDAD / AUSENCIA DE ALUCINACIONES (Groundedness / Faithfulness):
   - Compara la 'Respuesta Generada' ÚNICAMENTE con el 'Contexto Recuperado'. Cada afirmación de la respuesta debe estar respaldada por el contexto.
   - Justifica tu análisis brevemente en una frase.
   - Asigna una puntuación de 0.0 (toda la respuesta es inventada/conocimiento previo) a 1.0 (completamente respaldada por el contexto).

3. RELEVANCIA DE LA RESPUESTA (Answer Relevance):
   - Evalúa si la 'Respuesta Generada' responde de forma directa y clara a la 'Pregunta', ignorando el contexto. Evita rodeos o ambigüedades.
   - Justifica tu análisis brevemente en una frase.
   - Asigna una puntuación de 0.0 (no responde a lo que se pide) a 1.0 (respuesta perfecta y directa).

Devuelve el resultado ESTRICTAMENTE en este formato JSON, sin texto adicional antes ni después:
{{
  "context_relevance": {{
    "reasoning": "Tu justificación aquí",
    "score": 0.00
  }},
  "groundedness": {{
    "reasoning": "Tu justificación aquí",
    "score": 0.00
  }},
  "answer_relevance": {{
    "reasoning": "Tu justificación aquí",
    "score": 0.00
  }}
}}
""".strip()