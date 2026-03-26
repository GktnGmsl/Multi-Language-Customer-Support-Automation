from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class HFEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model(self):
        # Ragas sometimes expects a string for the model property in usage events
        return self.model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()
