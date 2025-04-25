import argparse
import os
import pandas as pd
from transformers import AutoTokenizer

from utils.blocking import block
from utils.dataset import AmazonDataset, GoogleDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel
from utils.index import get_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU Ids')
    args = parser.parse_args()
    batch_size = args.batch_size

    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data')

    print("Start blocking for batch size: ", batch_size)


    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    embedding_dim = 1024
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # embedding_dim = 384
    embedding_model = SentenceTransformerEmbeddingModel(model_name, device_ids=args.gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    amazon_dataset = AmazonDataset(os.path.join(data_path, "amazon_google/Amazon.csv"), tokenizer)
    google_dataset = GoogleDataset(os.path.join(data_path, "amazon_google/GoogleProducts.csv"), tokenizer)
    perfect_mapping_path = os.path.join(data_path, "amazon_google/Amzon_GoogleProducts_perfectMapping.csv")
    perfect_mapping_df = pd.read_csv(perfect_mapping_path)
    ground_truth = dict(zip(perfect_mapping_df['idAmazon'],perfect_mapping_df['idGoogleBase']))

    faiss_index = get_index(embedding_dim)

    block(google_dataset,
          amazon_dataset,
          embedding_model,
          faiss_index,
          batch_size,
          ground_truth,
          tokenizer,
          top_k=10,
          )
