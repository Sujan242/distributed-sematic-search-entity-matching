from torch.utils.data import DataLoader
import faiss
from utils.dataset import BaseDataset
from utils.embedding_model import EmbeddingModel
import torch
import faiss.contrib.torch_utils # need this for GPU support even though you don't use it

def build_index(dataset: BaseDataset, batch_size: int, embedding_model: EmbeddingModel, faiss_index, collator):

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collator)
    tableA_ids = []
    all_embeddings = []
    for batch in dataloader:
        ids = batch['id']
        embeddings = embedding_model.get_embedding(batch)
        all_embeddings.append(embeddings)
        tableA_ids.extend(ids)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = all_embeddings.contiguous()
    all_embeddings = all_embeddings.cpu().numpy()  # Move to CPU for FAISS
    faiss_index.train(all_embeddings)
    faiss_index.add(all_embeddings)
    return tableA_ids


def search_index(dataset: BaseDataset, batch_size: int,
                 embedding_model: EmbeddingModel, faiss_index,
                 top_k: int = 5,
                 tableA_ids: list = None,
                 collator=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collator)

    matches = {}

    for batch in dataloader:
        ids = batch['id']
        embeddings = embedding_model.get_embedding(batch) # TODO avoid CPU transfer here
        embeddings_np = embeddings.cpu().numpy()
        distances, indices = faiss_index.search(embeddings_np, top_k)

        for i, id in enumerate(ids):
            tableA_matches = [tableA_ids[idx] for idx in indices[i]]
            matches[id] = tableA_matches

    return matches
