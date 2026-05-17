from src.embedding import HuggingFaceEmbeddingsLC
from qdrant_client import QdrantClient as QdrantClientOfficial, models
from src.vector_store_qdrant import QdrantClient
from src.inferencer import GeminiInferencer
from prompts.generation import prompt as generation_prompt

class OnlinePipeline:
    def __init__(self, embedding_model_name, top_k, collection_name, llm_model_name):
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

    def do_rag(self, query) -> tuple[str, list[models.ScoredPoint]]:
        query_embedding = self.embedder.embed_query(query)

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True # to get also the text
        )

        context = "\n\n".join([result.payload["text"] for result in results.points])

        prompt = self.generation_prompt.format(context=context, query=query)

        text_response = self.inferencer.infer(prompt)
        return text_response, results.points