from torch.utils.data import DataLoader
import faiss
from utils.dataset import BaseDataset
from utils.embedding_model import EmbeddingModel
import faiss.contrib.torch_utils # need this for GPU support even though you don't use it

def build_index(dataset: BaseDataset, batch_size: int, embedding_model: EmbeddingModel, faiss_index):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    tableA_ids = []
    for batch in dataloader:
        ids = batch['id']
        sentences = batch['text']
        embeddings = embedding_model.get_embedding(sentences)
        faiss_index.add(embeddings)
        tableA_ids.extend(ids)
    return tableA_ids


def search_index(dataset: BaseDataset, batch_size: int,
                 embedding_model: EmbeddingModel, faiss_index,
                 top_k: int = 5,
                 tableA_ids: list = None):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    matches = {}

    for batch in dataloader:
        ids = batch['id']
        sentences = batch['text']

        embeddings = embedding_model.get_embedding(sentences)

        distances, indices = faiss_index.search(embeddings, top_k)

        for i, id in enumerate(ids):
            tableA_matches = [tableA_ids[idx] for idx in indices[i]]
            matches[id] = tableA_matches

    return matches
