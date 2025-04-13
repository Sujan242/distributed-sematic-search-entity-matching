import faiss
import torch


def get_index(dim):
    if torch.cuda.is_available():
        print("Using GPU for FAISS index")
        res = faiss.StandardGpuResources()
        return faiss.GpuIndexFlatL2(res, dim)

    print("Using CPU for FAISS index")
    return faiss.IndexFlatL2(dim)