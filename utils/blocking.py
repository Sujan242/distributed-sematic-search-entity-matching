import time

import pandas as pd

from utils.index_utils import build_index, search_index
from utils.evaluate_utils import evaluate

def block(first_dataset, second_dataset, embedding_model, faiss_index, batch_size, ground_truth, top_k, type="amazon-google"):
    blocking_start = time.time()
    print("Start building index...")
    build_start_time = time.time()
    tableA_ids = build_index(first_dataset, batch_size, embedding_model, faiss_index)
    build_end_time = time.time()
    index_search_start_time = time.time()
    print("Start searching...")
    # de
    # search index for table-B
    matches, matchData = search_index(dataset=second_dataset,
                           batch_size=batch_size,
                           embedding_model=embedding_model,
                           faiss_index=faiss_index,
                           top_k=top_k,
                           tableA_ids=tableA_ids,
                           type=type
                           )
    index_search_end_time = time.time()
    blocking_end = time.time()
    print("Build Index time: ", build_end_time - build_start_time)
    print("Index search time: ", index_search_end_time - index_search_start_time)
    print("Blocking time: ", blocking_end - blocking_start)
    # evaluate the results
    get_match_details(second_dataset, first_dataset, matchData, type=type)
    evaluate(matches, ground_truth, type=type)
    print("Candidate size = ", len(second_dataset)*top_k)

def get_match_details(tableA_dataset, tableB_dataset, match_data, type):
    match_df = pd.DataFrame(match_data)

    if type == "amazon-google":
        amazon_dataset_df = tableA_dataset.df
        amazon_df = amazon_dataset_df[['id', 'title', 'description']].copy()
        amazon_df.rename(columns={
            'title': 'titleAmazon',
            'description': 'descAmazon'
        }, inplace=True)

        google_dataset_df = tableB_dataset.df
        google_df = google_dataset_df[['id', 'name', 'description']].copy()
        google_df.rename(columns={
            'id': 'matchId',
            'name': 'titleGoogle',
            'description': 'descGoogle'
        }, inplace=True)

        augmented_df = match_df.merge(amazon_df, left_on='id', right_on='id', how='left')
        augmented_df = augmented_df.merge(google_df, left_on='matchId', right_on='matchId', how='left')

        textColumns = ['titleGoogle', 'descGoogle', 'titleAmazon', 'descAmazon']
        augmented_df[textColumns] = augmented_df[textColumns].fillna('#N/A')

        augmented_df.to_csv("./out-data/match_details_top10.csv", index=False)
    elif type == "songs":
        songs1_dataset_df = tableA_dataset.df
        songs1_df = songs1_dataset_df[['id', 'title', 'release', 'artist_name', 'duration', 'year']].copy()
        songs1_df.rename(columns={
            'id': 'idSong1',
            'title': 'titleSong1',
            'release': 'releaseSong1',
            'artist_name': 'artistSong1',
            'duration': 'durationSong1',
            'year': 'yearSong1'
        }, inplace=True)

        songs2_dataset_df = tableB_dataset.df
        songs2_df = songs2_dataset_df[['id', 'title', 'release', 'artist_name', 'duration', 'year']].copy()
        songs2_df.rename(columns={
            'id': 'idSong2',
            'title': 'titleSong2',
            'release': 'releaseSong2',
            'artist_name': 'artistSong2',
            'duration': 'durationSong2',
            'year': 'yearSong2'
        }, inplace=True)

        augmented_df = match_df.merge(songs1_df, left_on='id', right_on='idSong1', how='left')
        augmented_df = augmented_df.merge(songs2_df, left_on='matchId', right_on='idSong2', how='left')

        augmented_df.to_csv("./out-data/songs_details_top10.csv", index=False)

    return