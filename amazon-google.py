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
    parser.add_argument('--topk', type=int, default=10, help='Top k for faiss retrieval')
    parser.add_argument('--model', type=str, default='Alibaba-NLP/gte-large-en-v1.5', help='Model name for embedding')
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--use_fp16', action='store_true', help='Use fp16 for embedding model')
    parser.add_argument("--enable_profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()
    batch_size = args.batch_size

    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data')

    print(f"Start blocking for batch size:{batch_size}, gpus: {args.gpus}, topk: {args.topk}, model: {args.model}, embedding_dim: {args.embedding_dim}, use_fp16: {args.use_fp16}")

    embedding_model = SentenceTransformerEmbeddingModel(args.model, device_ids=args.gpus, use_fp16=args.use_fp16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    amazon_dataset = AmazonDataset(os.path.join(data_path, "amazon_google/Amazon.csv"), tokenizer)
    google_dataset = GoogleDataset(os.path.join(data_path, "amazon_google/GoogleProducts.csv"), tokenizer)
    perfect_mapping_path = os.path.join(data_path, "amazon_google/Amzon_GoogleProducts_perfectMapping.csv")
    perfect_mapping_df = pd.read_csv(perfect_mapping_path)
    ground_truth = dict(zip(perfect_mapping_df['idAmazon'],perfect_mapping_df['idGoogleBase']))

    faiss_index = get_index(args.embedding_dim)

    block(google_dataset,
          amazon_dataset,
          embedding_model,
          faiss_index,
          batch_size,
          ground_truth,
          tokenizer,
          args.topk,
          args.gpus,
          args.enable_profile
          )
