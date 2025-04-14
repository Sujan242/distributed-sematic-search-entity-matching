import argparse
import os
import time
import pandas as pd

from utils.dataset import AmazonDataset, GoogleDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel
from utils.evaluate_utils import evaluate
from utils.index import get_index
from utils.index_utils import build_index, search_index

if __name__ == "__main__":

    blocking_start = time.time()
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

    # build index for table-A
    embedding_model = SentenceTransformerEmbeddingModel("BAAI/bge-large-en-v1.5")
    faiss_index = get_index(1024)

    print("Start building index...")
    build_start_time = time.time()
    tableA_ids = build_index(amazon_dataset, batch_size, embedding_model, faiss_index)
    build_end_time = time.time()

    index_search_start_time = time.time()

    print("Start searching...")

    # de
    # search index for table-B
    matches = search_index(dataset=google_dataset,
                           batch_size=batch_size,
                           embedding_model=embedding_model,
                           faiss_index=faiss_index,
                           top_k=10,
                           tableA_ids=tableA_ids
    )
    index_search_end_time = time.time()

    blocking_end = time.time()
    print("Build Index time: ", build_end_time - build_start_time)
    print("Index search time: ", index_search_end_time - index_search_start_time)
    print("Blocking time: ", blocking_end - blocking_start)

    # evaluate the results
    perfect_mapping_path = os.path.join(data_path, "amazon_google/Amzon_GoogleProducts_perfectMapping.csv")
    perfect_mapping_df = pd.read_csv(perfect_mapping_path)
    ground_truth = dict(zip(perfect_mapping_df['idGoogleBase'], perfect_mapping_df['idAmazon']))
    evaluate(matches, ground_truth)
