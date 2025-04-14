from abc import abstractmethod
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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
        else:
            print("Using CPU for embedding")
            self.device = 'cpu'

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_embedding(self, sentences: List[str]):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)