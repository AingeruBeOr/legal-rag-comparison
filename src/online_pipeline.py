from src.embedding import HuggingFaceEmbeddingsLC
from qdrant_client import QdrantClient as QdrantClientOfficial, models
from src.vector_store_qdrant import QdrantClient
from src.inferencer import GeminiInferencer
from prompts.generation import prompt as generation_prompt
from sentence_transformers import CrossEncoder


class OnlinePipeline:
    def __init__(
        self,
        embedding_model_name,
        top_k,
        collection_name,
        llm_model_name,
        reranker_model_name=None,
        candidate_k=None,
    ):
        if llm_model_name == "gemini-3.1-flash-lite":
            self.inferencer = GeminiInferencer(llm_model_name)
        else:
            raise ValueError(f"Unsupported inferencer: {llm_model_name}")

        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.collection_name = collection_name
        collection_info = self.qdrant_client.get_collection_info(collection_name)
        collection_embedding_model = collection_info.config.metadata.get('embedding_model')

        if collection_embedding_model != embedding_model_name:
            raise ValueError(f"Embedding model mismatch: collection was created with {collection_embedding_model} but {embedding_model_name} was provided for RAG.")
        else:
            self.embedder = HuggingFaceEmbeddingsLC(model_name=embedding_model_name, show_progress=False)

        self.generation_prompt = generation_prompt
        self.top_k = top_k

        # --- Caso 3: retrieve-then-rerank ---
        # Si se pasa un reranker_model_name, primero se recupera un set amplio
        # de candidatos (candidate_k) con vector search, y luego un cross-encoder
        # los reordena por relevancia real query-documento, quedándose solo con
        # los top_k finales que entran al prompt.
        self.reranker_model_name = reranker_model_name
        self.reranker = CrossEncoder(reranker_model_name) if reranker_model_name else None
        # por defecto, si hay reranker, recuperamos 5x top_k candidatos antes de reordenar
        self.candidate_k = candidate_k if candidate_k is not None else self.top_k * 5

    def do_rag(self, query) -> tuple[str, list[models.ScoredPoint], dict]:
        query_embedding = self.embedder.embed_query(query)

        # Paso 1: recuperación. Con reranker activo, pedimos candidate_k
        # candidatos (más amplio); sin reranker, pedimos directamente top_k
        # (comportamiento original, caso 1: vector search puro).
        retrieval_limit = self.candidate_k if self.reranker else self.top_k

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=retrieval_limit,
            with_payload=True  # to get also the text
        )

        points = results.points

        # Paso 2: reranking de los candidatos con un cross-encoder.
        if self.reranker:
            pairs = [[query, point.payload["text"]] for point in points]
            rerank_scores = self.reranker.predict(pairs)

            reranked = sorted(
                zip(points, rerank_scores),
                key=lambda pair: pair[1],
                reverse=True,
            )[:self.top_k]

            # Sobrescribimos el score original (similitud coseno del vector
            # search) por el score del reranker, para que quede registrado
            # el criterio que realmente decidió el orden final.
            points = []
            for point, score in reranked:
                point.score = float(score)
                points.append(point)

        context = "\n\n".join([point.payload["text"] for point in points])

        prompt = self.generation_prompt.format(context=context, query=query)

        text_response, used_tokens = self.inferencer.infer(prompt)
        return text_response, points, used_tokens
        
        """
        from qdrant_client import QdrantClient as QdrantClientBase
        from fastembed import SparseTextEmbedding
        
        adalt va aquest
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        dense_embedding = self.dense_model.embed_query(query)
        sparse_embedding = next(self.sparse_model.embed([query]))
        
        results = self.qdrant_client.query_points(
        collection_name=self.collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_embedding,
                using="dense",
                limit=self.top_k * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(
                        indices=sparse_embedding.indices.tolist(),
                        values=sparse_embedding.values.tolist(),
                    ),
                using="sparse",
                limit=self.top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=None,  # If you don't want any filters for now
        limit=self.top_k,  # 5 the closest results
        with_payload=True # to get also the text
    )"""