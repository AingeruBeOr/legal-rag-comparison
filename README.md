# Comparativa de soluciones basadas en sistemas RAG sobre documentación legal

## Data

Dataset availability: TBA (Hugginface Datasets)

(https://alumnosunir-my.sharepoint.com/my?id=%2Fpersonal%2Faingeru%5Fbellido984%5Fcomunidadunir%5Fnet%2FDocuments%2FTFM%2FDataset&viewid=58c26454%2Dedbb%2D4630%2Dbdd5%2D2c69555beb97)

## Develop

1. Create a `.env` file, a copy from `.env.example`.
2. Create a virtual environment.
3. `pip install -r requirements.txt`.
4. Run Qdrant in Docker.
5. Documents into `data/raw/`

## Run experiments

1. Ingest documents: [ingest.py](scripts/ingest.py)
2. Do RAG for all the test set: [do_rag_test_set.py](scripts/do_rag_test_set.py)
3. Evaluate: [run_evaluation_on_results.py](evaluation/run_evaluation_on_results.py)
4. Get final results (averages and desvest): [run_get_final_results.py](evaluation/run_get_final_results.py)

## Components

## External Services

- LLM providers:
  - Google AI Studio: 
  - Groq

### Vector DB

Using Qdrant. Options:
- in-memory
- on disk
- Docker: 
  - start: `docker run -p 6333:6333 qdrant/qdrant`
  - UI: http://localhost:6333/dashboard

### Evaluation

LLM-as-a-judge evaluator: qwen/qwen3-32b (via Groq)

Metrics:
- LLM-as-a-judge:
  - Context Relevance
  - Groundedness / Faithfulness
  - Answer Relevance
- 
