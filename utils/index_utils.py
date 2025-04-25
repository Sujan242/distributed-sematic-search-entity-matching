from torch.utils.data import DataLoader
import faiss
from utils.dataset import BaseDataset
from utils.embedding_model import EmbeddingModel
import torch
import faiss.contrib.torch_utils # need this for GPU support even though you don't use it
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

def build_index(dataset: BaseDataset, batch_size: int, embedding_model: EmbeddingModel, faiss_index, collator, enable_profile):

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collator)
    tableA_ids = []
    all_embeddings = []

    if enable_profile:
        profiler_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True)
    else:
        profiler_context = nullcontext()

    with profiler_context as prof:

        for batch in dataloader:
            ids = batch['id']
            with record_function("EmbeddingModel:get_embedding"):
                embeddings = embedding_model.get_embedding(batch)  # runs on multiple GPUs

            with record_function("Collect Embeddings"):
                all_embeddings.append(embeddings)
                tableA_ids.extend(ids)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.contiguous()

        # Build FAISS index (assume faiss_index is on one GPU)
        with record_function("FAISS:train"):
            faiss_index.train(all_embeddings.cpu().numpy())  # move to CPU because FAISS expects numpy
        with record_function("FAISS:add"):
            faiss_index.add(all_embeddings.cpu().numpy())

    if enable_profile:
        print("Build profiling results")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    return tableA_ids


def search_index(dataset: BaseDataset, batch_size: int,
                 embedding_model: EmbeddingModel, faiss_index,
                 top_k: int = 5,
                 tableA_ids: list = None,
                 collator=None,
                 enable_profile=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collator)

    matches = {}

    if enable_profile:
        profiler_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True)
    else:
        profiler_context = nullcontext()

    with profiler_context as prof:
        for batch in dataloader:
            ids = batch['id']
            with record_function("EmbeddingModel:get_embedding"):
                embeddings = embedding_model.get_embedding(batch)  # runs on multiple GPUs

            with record_function("Embedding:move_to_CPU"):
                embeddings = embeddings.cpu()

            with record_function("FAISS:search"):
                distances, indices = faiss_index.search(embeddings.numpy(), top_k)

            with record_function("Postprocessing:match ids"):
                for i, id in enumerate(ids):
                    tableA_matches = [tableA_ids[idx] for idx in indices[i]]
                    matches[id] = tableA_matches

    if enable_profile:
        print("Search profiling results")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    return matches

