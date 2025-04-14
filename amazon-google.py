import argparse
import os
import pandas as pd

from utils.blocking import block
from utils.dataset import AmazonDataset, GoogleDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel
from utils.index import get_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    args = parser.parse_args()
    batch_size = args.batch_size

    print("Start blocking for batch size: ", batch_size)

    # load Table A
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data')

    amazon_dataset = AmazonDataset(os.path.join(data_path, "amazon_google/Amazon.csv"))
    google_dataset = GoogleDataset(os.path.join(data_path, "amazon_google/GoogleProducts.csv"))
    perfect_mapping_path = os.path.join(data_path, "amazon_google/Amzon_GoogleProducts_perfectMapping.csv")
    perfect_mapping_df = pd.read_csv(perfect_mapping_path)
    ground_truth = dict(zip(perfect_mapping_df['idAmazon'],perfect_mapping_df['idGoogleBase']))

    # build index for table-A
    embedding_model = SentenceTransformerEmbeddingModel("BAAI/bge-large-en-v1.5")
    faiss_index = get_index(1024)

    block(google_dataset,
          amazon_dataset,
          embedding_model,
          faiss_index,
          batch_size,
          ground_truth,
          top_k=10
          )
