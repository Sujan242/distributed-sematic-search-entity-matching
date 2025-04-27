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



def get_index(dim_in, use_pca = False, dim_out=None):
    """
    Create a FAISS index with optional PCA reduction.
    
    Args:
        dim_in (int): Original input embedding dimension (e.g., 1024).
        dim_out (int, optional): Dimension after PCA. If None, no PCA is applied.

    Returns:
        index (faiss.Index): FAISS index, possibly wrapped with PCA.
    """
    if dim_out is None:
        dim_out = dim_in  # No dimensionality reduction if not specified

    if use_pca:
        # Add PCA reduction
        print(f"Creating FAISS index with PCA: {dim_in} -> {dim_out}")
        pca = faiss.PCAMatrix(dim_in, dim_out)
        flat_index = faiss.IndexFlatIP(dim_out)
        cpu_index = faiss.IndexPreTransform(pca, flat_index)
    else:
        # No PCA needed
        cpu_index = faiss.IndexFlatIP(dim_out)
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU for FAISS index with dimension {dim_out}")
        res = faiss.StandardGpuResources()

        # Move index to GPU
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print(f"Using CPU for FAISS index with dimension {dim_out}")
        index = cpu_index

    return index