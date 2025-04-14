import time
from utils.index_utils import build_index, search_index
from utils.evaluate_utils import evaluate

def block(first_dataset, second_dataset, embedding_model, faiss_index, batch_size, ground_truth, top_k):
    blocking_start = time.time()
    print("Start building index...")
    build_start_time = time.time()
    tableA_ids = build_index(first_dataset, batch_size, embedding_model, faiss_index)
    build_end_time = time.time()
    index_search_start_time = time.time()
    print("Start searching...")
    # de
    # search index for table-B
    matches = search_index(dataset=second_dataset,
                           batch_size=batch_size,
                           embedding_model=embedding_model,
                           faiss_index=faiss_index,
                           top_k=top_k,
                           tableA_ids=tableA_ids
                           )
    index_search_end_time = time.time()
    blocking_end = time.time()
    print("Build Index time: ", build_end_time - build_start_time)
    print("Index search time: ", index_search_end_time - index_search_start_time)
    print("Blocking time: ", blocking_end - blocking_start)
    # evaluate the results
    evaluate(matches, ground_truth)