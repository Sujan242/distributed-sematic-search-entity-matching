import time

import faiss

from utils.index_utils import build_index, search_index
from utils.evaluate_utils import evaluate

from transformers import DataCollatorWithPadding
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

class CollatorWithID:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        ids = [f.pop('id') for f in features]  # remove 'id' before passing to tokenizer
        batch = self.data_collator(features)   # collate padded tensors
        batch['id'] = ids                      # reattach ids
        return batch

def block(first_dataset, second_dataset, embedding_model, faiss_index, batch_size, ground_truth, tokenizer, top_k, gpus, enable_profile):
    blocking_start = time.time()
    collator = CollatorWithID(tokenizer=tokenizer)
    print("Start building index...")
    build_start_time = time.time()
    tableA_ids = build_index(first_dataset, batch_size, embedding_model, faiss_index, collator, profile)
    build_end_time = time.time()
    index_search_start_time = time.time()

    # if multiple GPUs are available, replicate index across GPUs
    if torch.cuda.is_available() and len(gpus) > 1:
        if enable_profile:
            profiler_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                       record_shapes=True,
                                       profile_memory=True,
                                       with_stack=True)
        else:
            profiler_context = nullcontext()

        with profiler_context as prof:
            with record_function("Replication of index"):
                cloner_options = faiss.GpuMultipleClonerOptions()
                cloner_options.shard = False
                faiss_cpu_index = faiss.index_gpu_to_cpu(faiss_index)
                faiss_index = faiss.index_cpu_to_gpus_list(faiss_cpu_index, gpus=gpus, co=cloner_options)

        if enable_profile:
            print("Replication profiling results")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("Start searching...")
    # search index for table-B
    matches = search_index(dataset=second_dataset,
                           batch_size=batch_size,
                           embedding_model=embedding_model,
                           faiss_index=faiss_index,
                           top_k=top_k,
                           tableA_ids=tableA_ids,
                           collator=collator,
                           enable_profile=enable_profile
                           )
    index_search_end_time = time.time()
    blocking_end = time.time()
    print("Build Index time: ", build_end_time - build_start_time)
    print("Index search time: ", index_search_end_time - index_search_start_time)
    print("Blocking time: ", blocking_end - blocking_start)
    # evaluate the results
    evaluate(matches, ground_truth)
    print("Candidate size = ", len(second_dataset)*top_k)