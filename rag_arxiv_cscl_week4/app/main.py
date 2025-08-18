from fastapi import FastAPI
import json
import numpy as np
import faiss
import os
from rag.embed_utils import Embedder

app = FastAPI()

# Define data directory
data_dir = "data"

#加载数据
with open(f"{data_dir}/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line)["text"] for line in f]
embeddings  = np.load(f"{data_dir}/embeddings.npy")
faiss_index = faiss.read_index(f"{data_dir}/faiss.index")

embedder = Embedder()

@app.get("/search")
async def search(q: str):
    query_vec = embedder.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(query_vec, 3)
    results = [chunks[i] for i in I[0]]
    return {"query": q, "results": results}