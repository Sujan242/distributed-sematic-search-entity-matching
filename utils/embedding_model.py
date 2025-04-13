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
        if torch.cuda.is_available():
            print(f"Using GPU for embedding: {torch.cuda.get_device_name(0)}")
            self.device = 'cuda'
            print("Using device: ", self.device)
        else:
            print("Using CPU for embedding")
            self.device = 'cpu'

        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embedding(self, sentences: List[str]):
        # TODO split across GPUs
        return self.model.encode(sentences,
                                 convert_to_tensor=True,
                                 batch_size=len(sentences),
                                 device=self.device
                                 )