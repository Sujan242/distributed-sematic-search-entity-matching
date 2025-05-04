from torch.utils.data import DataLoader
import faiss
from utils.dataset import BaseDataset
from utils.embedding_model import EmbeddingModel
import torch
import faiss.contrib.torch_utils # need this for GPU support even though you don't use it

def build_index(dataset: BaseDataset, batch_size: int, embedding_model: EmbeddingModel, faiss_index, collator, nprobe = 100):

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collator)
    tableA_ids = []
    # all_embeddings = []
    initial_embeddings = []
    all_emb_batches = []

    # 1) Embed everything and collect IDs
    for batch in dataloader:
        ids       = batch['id']
        emb_torch = embedding_model.get_embedding(batch)     # on GPU (if you drop .cpu())
        emb_cpu   = emb_torch.detach().cpu()                 # move to host
        tableA_ids.extend(ids)
        all_emb_batches.append(emb_cpu)

    # 2) Concatenate into one big array
    all_embeddings = torch.cat(all_emb_batches, dim=0).numpy().astype('float32')
    # shape = (N_total, embedding_dim)

    # 3) Train on *all* embeddings
    faiss_index.train(all_embeddings)
    print(f"Trained IVF on {all_embeddings.shape[0]} vectors")
    # 4) Add all embeddings to the index

    faiss_index.add(all_embeddings)
    print(f"Added {all_embeddings.shape[0]} vectors to index")
    faiss_index.nprobe = 50
    # for batch in dataloader:
    #     ids = batch['id']
    #     embeddings = embedding_model.get_embedding(batch)
    #     if faiss_index.is_trained:
    #         faiss_index.add(embeddings)
    #     else:
    #         if len(initial_embeddings) <= (80000/batch_size):
    #             initial_embeddings.append(embeddings)
    #         else:
    #             initial_embeddings = torch.cat(initial_embeddings, dim=0)
    #             faiss_index.train(initial_embeddings)
    #             print("trained faiss index")
        # all_embeddings.append(embeddings)
    #     tableA_ids.extend(ids)
    #
    # all_embeddings = torch.cat(all_embeddings, dim=0)
    # all_embeddings = all_embeddings.contiguous()
    print("Done building index")
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
        embeddings = embedding_model.get_embedding(batch).cpu() # TODO avoid CPU transfer here

        distances, indices = faiss_index.search(embeddings, top_k)

        for i, id in enumerate(ids):
            tableA_matches = [tableA_ids[idx] for idx in indices[i]]
            matches[id] = tableA_matches
            print(f"ID: {id}, Matches: {tableA_matches}")

    return matches
