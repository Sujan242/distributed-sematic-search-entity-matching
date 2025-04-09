import os
import argparse

from torch.utils.data import DataLoader

from utils.dataset import AmazonDataset, BaseDataset, GoogleDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel, EmbeddingModel
import time
import faiss
from faiss import Index


def build_index(dataset: BaseDataset, batch_size: int, embedding_model: EmbeddingModel, faiss_index: Index):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        ids = batch['id']
        sentences = batch['text']
        embeddings = embedding_model.get_embedding(sentences)
        faiss_index.add(embeddings.cpu().numpy())


if __name__ == "__main__":

    blocking_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    args = parser.parse_args()
    batch_size = args.batch_size

    # load Table A
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data')
    amazon_dataset = AmazonDataset(os.path.join(data_path, "amazon_google/Amazon.csv"))
    google_dataset = GoogleDataset(os.path.join(data_path, "amazon_google/GoogleProducts.csv"))

    # build index for table-A
    embedding_model = SentenceTransformerEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = faiss.IndexFlatL2(384)

    build_start_time = time.time()
    build_index(amazon_dataset, batch_size, embedding_model, faiss_index)
    build_end_time = time.time()




    blocking_end = time.time()
    print("Build time: ", build_end_time - build_start_time)
    print("Blocking time: ", blocking_end - blocking_start)