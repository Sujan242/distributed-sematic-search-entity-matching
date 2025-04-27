import csv

from torch.utils.data import DataLoader
import faiss
from utils.dataset import BaseDataset
from utils.embedding_model import EmbeddingModel
import torch
import faiss.contrib.torch_utils # need this for GPU support even though you don't use it

def build_index(dataset: BaseDataset, batch_size: int, embedding_model: EmbeddingModel, faiss_index):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    tableA_ids = []
    all_embeddings = []
    for batch in dataloader:
        ids = batch['id']
        sentences = batch['text']
        embeddings = embedding_model.get_embedding(sentences)
        all_embeddings.append(embeddings)
        tableA_ids.extend(ids)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = all_embeddings.contiguous()
    faiss_index.train(all_embeddings)
    faiss_index.add(all_embeddings)
    return tableA_ids


def search_index(dataset: BaseDataset, batch_size: int,
                 embedding_model: EmbeddingModel, faiss_index,
                 top_k: int = 5,
                 tableA_ids: list = None, type='amazon-google'):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    matches = {}
    matchData = []

    for batch in dataloader:
        ids = batch['id']
        # print(f"DEBUG: shape(ids)={len(ids)}")
        sentences = batch['text']

        embeddings = embedding_model.get_embedding(sentences)

        distances, indices = faiss_index.search(embeddings, top_k)
        # print(f'DEBUG: shape(distances)={distances.shape}, shape(indices)={indices.shape}')

        for i, id in enumerate(ids):
            tableA_matches = [tableA_ids[idx] for idx in indices[i]]
            # print(f"DEBUG: --- i={i}, id={id}, tableA_matches={tableA_matches}")
            matches[id] = tableA_matches

            # print(f"DEBUG: shape(tableA_matches)={len(tableA_matches)}")

            for k in range(len(tableA_matches)):
                if type == 'amazon-google':
                    matchData.append({
                    'id': id,
                    'matchNum': k,
                    'matchIdx': int(indices[i][k]),
                    'distance': float(distances[i][k]),
                    'matchId': tableA_matches[k]
                    })
                    print(f"DEBUG: record={matchData[-1]}")
                elif type == 'songs':
                    matchData.append({
                        'id': id,
                        'matchNum': k,
                        'matchIdx': int(indices[i][k]),
                        'distance': float(distances[i][k]),
                        'matchId': tableA_matches[k]
                    })

        # fieldnames = ['id', 'matchNum', 'matchIdx', 'distance', 'matchId']
        #
        # with open('match_details.csv', 'w', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     writer.writerows(matchData)

    return matches, matchData