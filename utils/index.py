import faiss

def get_index(dim):
    return faiss.IndexFlatL2(dim)