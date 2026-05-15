from src.embedding import HuggingFaceEmbeddingsLC
from qdrant_client import QdrantClient, models
from src.inferencer import GeminiInferencer
from prompts.generation import prompt as generation_prompt

class OnlinePipeline:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddingsLC(show_progress=False)
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.inferencer = GeminiInferencer()

        self.generation_prompt = generation_prompt

    def do_rag(self, query):
        query_embedding = self.embedder.embed_query(query)

        results = self.qdrant_client.query_points(
            collection_name="documents",
            query=query_embedding,
            limit=5,
            with_payload=True # to get also the text
        )

        context = "\n\n".join([result.payload["text"] for result in results.points])

        prompt = self.generation_prompt.format(context=context, query=query)

        text_response = self.inferencer.infer(prompt)
        return text_response