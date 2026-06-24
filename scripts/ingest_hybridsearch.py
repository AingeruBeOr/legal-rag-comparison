import sys
import os
import uuid
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import torch
import gc
from fastembed import TextEmbedding, SparseTextEmbedding
sys.path.append("../src/")

# -- CONFIG --
class Config:
    DOCUMENTS_PATH = os.path.abspath("../legal-rag-comparison/data")
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    VECTOR_SIZE = 1024

    COLLECTION_NAME = "documents"
    
    # usamos batch size para optimizar la generación de embeddings
    BATCH_SIZE = 1 # con 4 ya me coge 12-13GB de RAM
    CHUNK_BATCH_SIZE = 32

os.chdir(r"c:\Users\Usuario\Desktop\RyP_4Pv3_Actividad_Laboratorio_Código_para_estudiantes\legal-rag-comparison\src")
print(os.getcwd())
from parser import PDFParser
from chunking import PDFChunker
from embedding import HuggingFaceEmbeddingsLC

parser = PDFParser()
chunker = PDFChunker()
dense_embedder = HuggingFaceEmbeddingsLC(show_progress=False)
sparse_embedder = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)


# -- PROCESS --
print(f"Documents path: {Config.DOCUMENTS_PATH}")
pdfs = [file for file in os.listdir(Config.DOCUMENTS_PATH) if file.endswith(".pdf") or file.endswith(".PDF")]
print(f"Número de documentos en la carpeta '{Config.DOCUMENTS_PATH}': {len(pdfs)}")

client = QdrantClient("http://localhost:6333")

if client.collection_exists(Config.COLLECTION_NAME):
    input(f"Collection '{Config.COLLECTION_NAME}' already exists. Are you sure you want to delete it and re-ingest?")
    client.delete_collection(Config.COLLECTION_NAME)

client.create_collection(
    collection_name=Config.COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=Config.VECTOR_SIZE,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
    metadata={
        "embedding_model": Config.EMBEDDING_MODEL,
        "vector_size": Config.VECTOR_SIZE
    }
)

def batch_iterator(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i:i + batch_size]

input("Ready to ingest documents. Press Enter to continue...")

for pdfs_batch in tqdm(batch_iterator(pdfs, Config.BATCH_SIZE), total=len(pdfs) // Config.BATCH_SIZE + 1):
    all_pdf_chunks = []
    for file in pdfs_batch:
        file_path = os.path.join(Config.DOCUMENTS_PATH, file)
        pdf_text = parser.parse(file_path)
        pdf_chunks = chunker.chunk(pdf_text)

        for chunk in pdf_chunks:
            meta = chunk.metadata.copy() if chunk.metadata else {}
            meta["source_file"] = file
            all_pdf_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk.page_content,
                "metadata": meta
            })

    for chunk_batch in batch_iterator(all_pdf_chunks, Config.CHUNK_BATCH_SIZE):
        texts = [item["text"] for item in chunk_batch]
        dense_embeddings = dense_embedder.embed_documents(texts)
        sparse_embeddings = list(sparse_embedder.embed(texts))

        # points for qdrant
        points = [
            models.PointStruct(
                id=item["id"],
                vector={
                "dense": dense_embeddings,
                "sparse": models.SparseVector(
                    indices=sparse_embeddings.indices.tolist(),
                    values=sparse_embeddings.values.tolist(),
                ),
            },
                payload={**item["metadata"], "text": item["text"]}
            )
            for item, dense_embeddings, sparse_embeddings  in zip(chunk_batch, dense_embeddings, sparse_embeddings)
        ]

        client.upsert(
            collection_name=Config.COLLECTION_NAME,
            points=points
        )

        # free memory
        del texts, dense_embeddings, sparse_embeddings, points
        gc.collect() 
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # free memory
    del all_pdf_chunks

print("Ingestion completed.")