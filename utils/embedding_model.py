from abc import abstractmethod
from sentence_transformers import SentenceTransformer
from typing import List
import torch

class EmbeddingModel:

    @abstractmethod
    def get_embedding(self, sentences: List[str]):
        """
        Get the embedding for the given sentences.

        Args:
            sentences (list): A list of sentences to get embeddings for (single batch).

        Returns:
            list: A list of embeddings for the given sentences.
        """
        pass

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, sentences: List[str]):
        # TODO split across GPUs
        return self.model.encode(sentences,
                                 convert_to_tensor=True,
                                 batch_size=len(sentences),
                                 device=self.device
                                 )