from langchain_huggingface import HuggingFaceEmbeddings
import torch

class HuggingFaceEmbeddingsLC:
    def __init__(self, model_name="BAAI/bge-m3", show_progress=True):
        self.model_name = model_name
        
        # check cuda availability
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for embeddings.")
        # cehck MPS availability (for Apple Silicon)
        elif torch.backends.mps.is_available():
            print("MPS is available. Using Apple Silicon GPU for embeddings.")
        else:
            print("CUDA and MPS are not available. Using CPU for embeddings.")

        self.embedder = HuggingFaceEmbeddings(model_name=model_name, show_progress=show_progress)

    def embed_query(self, text):
        return self.embedder.embed_query(text)
    
    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)
