import faiss
import torch


def get_index(dim, nlist=1000):
    if torch.cuda.is_available():
        print("Using GPU for FAISS index")
        res = faiss.StandardGpuResources()

        quantizer = faiss.IndexFlatIP(dim)  # Quantizer for IVF
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Move it to GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        return gpu_index
    else:
        print("Using CPU for FAISS index")
        index = faiss.IndexFlatIP(dim)
    return index