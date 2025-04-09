import os
import argparse

from torch.utils.data import DataLoader

from utils.dataset import AmazonDataset, BaseDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel
import time


def build_index(dataset: BaseDataset, batch_size: int, embedding_model: SentenceTransformerEmbeddingModel):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        sentences = batch['text']
        embeddings = embedding_model.get_embedding(sentences)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    args = parser.parse_args()
    batch_size = args.batch_size

    # load Table A
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data')
    amazon_dataset = AmazonDataset(os.path.join(data_path, "amazon_google/Amazon.csv"))

    # build index for table-A
    model = SentenceTransformerEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

    build_start_time = time.time()
    build_index(amazon_dataset, batch_size, model)
    build_end_time = time.time()


    print("Build time: ", build_end_time - build_start_time)