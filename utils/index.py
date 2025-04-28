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


def get_index(dim_in, index_type='flat', nlist=100, use_pca = False, dim_out=None):
    if index_type == 'flat':
        print("Using flat index")
        return create_flat_index(dim_in, use_pca, dim_out)
    elif index_type == 'ivf':
        print("Using IVF index")
        return create_ivf_index(dim_in, use_pca, dim_out, nlist)
    else:
        raise ValueError(f"Unknown index type: {index_type}. Supported types are 'flat' and 'ivf'.")


def create_flat_index(dim_in, use_pca = False, dim_out=None):
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

def create_ivf_index(dim_in, use_pca = False, dim_out=None, nlist=100):
    """
    Create a FAISS IVF index with optional PCA reduction.
    
    Args:
        dim_in (int): Original input embedding dimension (e.g., 1024).
        dim_out (int, optional): Dimension after PCA. If None, no PCA is applied.
        nlist (int): Number of clusters for IVF.

    Returns:
        index (faiss.Index): FAISS IVF index, possibly wrapped with PCA.
    """
    if dim_out is None:
        dim_out = dim_in  # No dimensionality reduction if not specified
    
    if use_pca:
        # Add PCA reduction
        print(f"Creating FAISS IVF index with PCA: {dim_in} -> {dim_out}")
        quantizer = faiss.IndexFlatL2(dim_out if use_pca else dim_in)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim_out if use_pca else dim_in, nlist, faiss.METRIC_L2)
        pca = faiss.PCAMatrix(dim_in, dim_out)
        cpu_index = faiss.IndexPreTransform(pca, ivf_index)
        if torch.cuda.is_available():
            print(f"Using GPU for FAISS IVF index with dimension {dim_out}")
            res = faiss.StandardGpuResources()
            # Move index to GPU
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        # No PCA needed
        print(f"Creating FAISS IVF index without PCA: {dim_in} -> {dim_out}")
        if torch.cuda.is_available():
            print(f"Using GPU for FAISS IVF index with dimension {dim_out}")
            resources = faiss.StandardGpuResources()
            index = faiss.GpuIndexIVFFlat(resources, dim_out, nlist, faiss.METRIC_L2)
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim_out), dim_out, nlist, faiss.METRIC_L2)
    return index