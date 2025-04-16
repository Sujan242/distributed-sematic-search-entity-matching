import faiss
import torch


def get_index(dim):
    if torch.cuda.is_available():
        print("Using GPU for FAISS index")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim)
    else:
        print("Using CPU for FAISS index")
        index = faiss.IndexFlatIP(dim)
    return index