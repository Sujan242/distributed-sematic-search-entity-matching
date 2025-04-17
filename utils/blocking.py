import time

from utils.index_utils import build_index, search_index
from utils.evaluate_utils import evaluate

from transformers import DataCollatorWithPadding

class CollatorWithID:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        ids = [f.pop('id') for f in features]  # remove 'id' before passing to tokenizer
        batch = self.data_collator(features)   # collate padded tensors
        batch['id'] = ids                      # reattach ids
        return batch

def block(first_dataset, second_dataset, embedding_model, faiss_index, batch_size, ground_truth, tokenizer, top_k):
    blocking_start = time.time()
    collator = CollatorWithID(tokenizer=tokenizer)
    print("Start building index...")
    build_start_time = time.time()
    tableA_ids = build_index(first_dataset, batch_size, embedding_model, faiss_index, collator)
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
                           tableA_ids=tableA_ids,
                           collator=collator
                           )
    index_search_end_time = time.time()
    blocking_end = time.time()
    print("Build Index time: ", build_end_time - build_start_time)
    print("Index search time: ", index_search_end_time - index_search_start_time)
    print("Blocking time: ", blocking_end - blocking_start)
    # evaluate the results
    evaluate(matches, ground_truth)
    print("Candidate size = ", len(second_dataset)*top_k)