import argparse
import os
import pandas as pd
from torch.nn.functional import embedding

from utils.blocking import block
from utils.dataset import SongsDataset
from utils.embedding_model import SentenceTransformerEmbeddingModel
from utils.index import get_index
from transformers import AutoTokenizer

import requests


def download_csv(url, local_filepath):
    """
    Downloads the MSD songs CSV file from the Wisconsin CS website and saves it as 'songs.csv'
    in the current directory.
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

    print("Start blocking for batch size: ", batch_size)

    # load Table A
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data/songs')

    missingSongsDir = not os.path.isdir(data_path)
    missingSongsFile = not os.path.isfile(os.path.join(data_path, 'songs.csv'))
    missingSongsGoldenFile = not os.path.isfile(os.path.join(data_path, 'Songs_perfectMapping.csv'))

    if missingSongsDir:
        os.makedirs(data_path, exist_ok=True)
    if missingSongsFile:
        download_csv(url="http://pages.cs.wisc.edu/~anhai/data/falcon_data/songs/msd.csv", local_filepath='./data/songs/songs.csv')
    if missingSongsGoldenFile:
        download_csv(url="http://pages.cs.wisc.edu/~anhai/data/falcon_data/songs/matches_msd_msd.csv", local_filepath='./data/songs/Songs_perfectMapping.csv')

    print(
        f"Start blocking for batch size:{batch_size}, gpus: {args.gpus}, topk: {args.topk}, model: {args.model}, embedding_dim: {args.embedding_dim}, use_fp16: {args.use_fp16}")

    # build index for table-A
    embedding_model = SentenceTransformerEmbeddingModel(args.model, device_ids=args.gpus, use_fp16=args.use_fp16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    songs1_dataset = SongsDataset(os.path.join(data_path, 'songs.csv'), tokenizer=tokenizer)
    print(f"Created dataset #1 of {len(songs1_dataset)} songs")
    songs2_dataset = SongsDataset(os.path.join(data_path, 'songs.csv'), tokenizer=tokenizer)
    print(f"Created dataset #2 of {len(songs2_dataset)} songs")
    perfect_mapping_path = os.path.join(data_path, "Songs_perfectMapping.csv")
    perfect_mapping_df = pd.read_csv(perfect_mapping_path)
    perfect_mapping_df.rename(columns={
        'id1': 'idSong1',
        'id2': 'idSong2',
    }, inplace=True)
    ground_truth = dict(zip(perfect_mapping_df['idSong1'],perfect_mapping_df['idSong2']))
    faiss_index = get_index(args.embedding_dim)


    block(songs1_dataset,
          songs2_dataset,
          embedding_model,
          faiss_index,
          batch_size,
          ground_truth,
          tokenizer,
          args.topk,
          args.gpus
          )