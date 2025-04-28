import faiss
# import faiss.contrib.gpu as faiss_gpu
import torch


def get_index(dim):
    if torch.cuda.is_available():
        print("Using GPU for FAISS index")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim) #faiss.GpuIndexIVFFlat(res, dim)
    else:
        # import faiss
        print("Using CPU for FAISS index")
        index = faiss.IndexFlatIP(dim)
    return index
