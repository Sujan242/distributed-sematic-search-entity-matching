import argparse
import os
import pandas as pd
from transformers import AutoTokenizer

from utils.blocking import block
from utils.dataset import AmazonDataset, GoogleDataset, NewAmazonDataset, WalmartDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel
from utils.index import get_index

import requests

def download_csv(url, local_filepath):
    """
    Downloads the CSV file from the Wisconsin CS website and saves it in the current directory.

    Returns:
        str: Path to the downloaded file if successful, None otherwise
    """
    try:
        # Send GET request to download the file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)

        # Check if the request was successful
        response.raise_for_status()

        # Get total file size for progress reporting
        total_size = int(response.headers.get('content-length', 0))

        # Write the file to disk
        with open(local_filepath, 'wb') as file:
            if total_size == 0:  # No content length header
                file.write(response.content)
                print(f"Downloaded file (unknown size)")
            else:
                downloaded = 0
                chunk_size = 8192  # 8KB chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        file.write(chunk)
                        downloaded += len(chunk)

                        # Print progress
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

                print("\nDownload complete!")

        # Verify the file exists and has content
        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) > 0:
            print(f"File saved as '{local_filepath}'")
            return local_filepath
        else:
            print("Error: Downloaded file is empty or not found")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU Ids')
    parser.add_argument('--topk', type=int, default=10, help='Top k for faiss retrieval')
    parser.add_argument('--model', type=str, default='Alibaba-NLP/gte-large-en-v1.5', help='Model name for embedding')
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--use_fp16', action='store_true', help='Use fp16 for embedding model')
    args = parser.parse_args()
    batch_size = args.batch_size

    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data/walmart_amazon')

    print("Loading data ...")
    missingDataDir = not os.path.isdir(data_path)
    missingWalmartFile = not os.path.isfile(os.path.join(data_path, 'walmart.csv'))
    missingAmazonFile = not os.path.isfile(os.path.join(data_path, 'amazon.csv'))
    missingGoldenFile = not os.path.isfile(os.path.join(data_path, 'walmart_amazon_perfectmapping.csv'))

    if missingDataDir:
        os.makedirs(data_path, exist_ok=True)
        print(f"Created missing data directory {data_path}")

    if missingWalmartFile:
        res = download_csv(url="http://pages.cs.wisc.edu/~anhai/data/corleone_data/products/walmart.csv", local_filepath='./data/walmart_amazon/walmart.csv')
    else:
        print(f"File already downloaded: {os.path.join(data_path, 'walmart.csv').replace(os.getcwd(), '.')}")
        
    with open(os.path.join(data_path, 'walmart.csv'), 'r') as f:
        lines = f.readlines()

    expectedSchema = 'id,product_id,upc,brand,category,title,price,shelfdescr,shortdescr,longdescr,imageurl,orig_shelfdescr,orig_shortdescr,orig_longdescr,modelno,shipweight,dimensions\n'
    if lines[0] != expectedSchema:
        lines[0] = expectedSchema

        with open(os.path.join(data_path, 'walmart.csv'), 'w') as f:
            f.writelines(lines)

    if missingAmazonFile:
        res = download_csv(url="http://pages.cs.wisc.edu/~anhai/data/corleone_data/products/amazon.csv", local_filepath='./data/walmart_amazon/amazon.csv')
    else:
        print(f"File already downloaded: {os.path.join(data_path, 'amazon.csv').replace(os.getcwd(), '.')}")
    
    with open(os.path.join(data_path, 'amazon.csv'), 'r') as f:
        lines = f.readlines()

    expectedSchema = 'id,url,asin,brand,modelno,category,pcategory1,category2,pcategory2,title,listprice,price,prodfeatures,techdetails,proddescrshort,proddescrlong,dimensions,imageurl,itemweight,shipweight,orig_prodfeatures,orig_techdetails\n'
    if lines[0] != expectedSchema:
        lines[0] = expectedSchema

        with open(os.path.join(data_path, 'amazon.csv'), 'w') as f:
            f.writelines(lines)

    if missingGoldenFile:
        download_csv(url="http://pages.cs.wisc.edu/~anhai/data/corleone_data/products/matches_walmart_amazon.csv", local_filepath='./data/walmart_amazon/walmart_amazon_perfectmapping.csv')
    else:
        print(f"File already downloaded: {os.path.join(data_path, 'walmart_amazon_perfectmapping.csv').replace(os.getcwd(), '.')}")

    print(f"Start blocking for batch size:{batch_size}, gpus: {args.gpus}, topk: {args.topk}, model: {args.model}, embedding_dim: {args.embedding_dim}, use_fp16: {args.use_fp16}")

    embedding_model = SentenceTransformerEmbeddingModel(args.model, device_ids=args.gpus, use_fp16=args.use_fp16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    walmartColumns = ["id","title","category","brand","modelno","price"]
    walmart_dataset = WalmartDataset(os.path.join(data_path, "walmart.csv"), tokenizer, columns=walmartColumns)

    amazonColumns = ["id","title","category","brand","modelno","price"]
    amazon_dataset = NewAmazonDataset(os.path.join(data_path, "amazon.csv"), tokenizer, columns=amazonColumns)

    # When evaluating dataset1=amazon, dataset2=walmart have ground_truth as id1:id2
    # When evaluating dataset1=walmart, dataset2=amazon have ground_truth as id2:id1
    perfect_mapping_path = os.path.join(data_path, "walmart_amazon_perfectmapping.csv")
    perfect_mapping_df = pd.read_csv(perfect_mapping_path)
    ground_truth = {} # dict(zip(perfect_mapping_df['id1'],perfect_mapping_df['id2']))
    for rNum, row in perfect_mapping_df.iterrows():
        srcLabel = 'id1'
        tgtLabel = 'id2'
        if row[srcLabel] in ground_truth.keys():
            row[srcLabel].append(row[tgtLabel])
        else:
            ground_truth[row[srcLabel]] = [row[tgtLabel]]

    faiss_index = get_index(args.embedding_dim)

    block(amazon_dataset,
          walmart_dataset,
          embedding_model,
          faiss_index,
          batch_size,
          ground_truth,
          tokenizer,
          args.topk,
          args.gpus
          )
