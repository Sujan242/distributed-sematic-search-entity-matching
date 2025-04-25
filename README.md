# Entity Matching with Sentence Transformers and FAISS

This project performs blocking for entity matching between any two datasets, using LLM embeddings and FAISS-based approximate nearest neighbor search.

---

## 1. Set up a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

---

## 2. Install Required Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements_gpu.txt
```

---

## 3. Prepare the Dataset

Ensure the following files exist under `data/amazon_google/`:

```
data/amazon_google/Amazon.csv
data/amazon_google/GoogleProducts.csv
data/amazon_google/Amzon_GoogleProducts_perfectMapping.csv
```

> 📂 Example Project Structure:
> ```
> your_project/
> ├── data/
> │   └── amazon_google/
> │       ├── Amazon.csv
> │       ├── GoogleProducts.csv
> │       └── Amzon_GoogleProducts_perfectMapping.csv
> ├── amazon-google.py
> ├── requirements_gpu.txt
> ```

---

## 4. Run the Program

```bash
python amazon-google.py \
    --batch_size 32 \
    --gpus 0 1 \
    --topk 10 \
    --model Alibaba-NLP/gte-large-en-v1.5 \
    --embedding_dim 1024 \
    --use_fp16
```

---

## 5. Argument Descriptions

| Argument | Description | Default                           |
|:---------|:------------|:----------------------------------|
| `--batch_size` | Batch size for embedding sentences | `1`                               | `32` |
| `--gpus` | GPU IDs to use (space-separated) | `0`                               | `0 1` |
| `--topk` | Number of top candidates retrieved per query | `10`                              | `10` |
| `--model` | Huggingface model name for embeddings | `'Alibaba-NLP/gte-large-en-v1.5'` | `'Alibaba-NLP/gte-large-en-v1.5'` |
| `--embedding_dim` | Dimension of embeddings (depends on model) | `1024`                            | `1024` |
| `--use_fp16` | (Flag) Use half-precision (fp16) inference | `False`                           | (just include the flag) |

> ⚠️ Note: For `--use_fp16`, you do not pass a value — just add `--use_fp16` to enable it.

---

## 6. Notes

- Ensure your system has CUDA-enabled GPUs if you want to use `--gpus` argument.
- If `--use_fp16` is enabled, embeddings will be computed faster with lower memory usage, but final embeddings must be cast back to float32 before FAISS search.

---
