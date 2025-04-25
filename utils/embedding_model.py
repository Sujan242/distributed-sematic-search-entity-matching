from abc import abstractmethod
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class EmbeddingModel:

    @abstractmethod
    def get_embedding(self, encoded_input):
        """
        Get the embedding for the given sentences.

        Args:
            sentences (list): A list of sentences to get embeddings for (single batch).

        Returns:
            list: A list of embeddings for the given sentences.
        """
        pass

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, device_ids: List[int] = None, use_fp16: bool = False):
        if torch.cuda.is_available():
            print(f"Using GPU for embedding: {torch.cuda.get_device_name(0)}")
            self.device = 'cuda'
            self.device_ids = device_ids
        else:
            print("Using CPU for embedding")
            self.device = 'cpu'

        self.model = AutoModel.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          torch_dtype=torch.float16 if use_fp16 else torch.float32)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.model = self.model.to(self.device)



    def get_embedding(self, encoded_input):
        """
        Reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        """
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)