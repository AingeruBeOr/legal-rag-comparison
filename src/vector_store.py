from tqdm import tqdm
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.embeddings import Embeddings

class LangchainEmbedderWrapper(Embeddings):
    """Adapter to make the custom GeminiEmbedder compatible with LangChain Embeddings interface."""
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # with tqdm
        return [self.embedder.embed(t)[0].values for t in tqdm(texts, desc="Embedding documents")]

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed(text)[0].values

def get_qdrant_vector_store(embedder, collection_name="documents") -> QdrantVectorStore:
    """Returns an instance of Langchain QdrantVectorStore in-memory."""
    lc_embedder = LangchainEmbedderWrapper(embedder)
    client = QdrantClient(":memory:")

    # create collection if it doesn't exist
    if collection_name not in client.get_collections().collections:
        client.recreate_collection(collection_name=collection_name, 
                                   vectors_config=models.VectorParams(size=embedder.vector_size, distance=models.Distance.COSINE))
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=lc_embedder,
    )