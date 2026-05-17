from qdrant_client import QdrantClient as QdrantClientBase

class QdrantClient:
    def __init__(self, url="http://localhost:6333"):
        self.qdrant_client = QdrantClientBase(url)

    def query_points(self, collection_name, query, limit=5, with_payload=True):
        results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query,
            limit=limit,
            with_payload=with_payload
        )
        return results
    
    def get_collection_info(self, collection_name):
        return self.qdrant_client.get_collection(collection_name)