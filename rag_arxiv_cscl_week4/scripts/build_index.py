import json
import os
import numpy as np
import faiss
from rag.pdf_utils import extract_text_from_pdf
from rag.text_utils import chunk_text
from rag.embed_utils import Embedder
from rag.index_utils import build_faiss_index

data_dir = "data"
meta_file = os.path.join(data_dir, "meta.jsonl")
chunks = []
metas = []

# 1. 遍历 meta.jsonl
with open(meta_file, "r", encoding="utf-8") as f:
    for line in f:
        meta = json.loads(line)
        text = extract_text_from_pdf(meta["pdf_path"])
        paper_chunks = chunk_text(text)
        chunks.extend(paper_chunks)
        metas.extend([meta]*len(paper_chunks))

print(f"Total chunks: {len(chunks)}")

# 2. 嵌入
embedder = Embedder()
embeddings = embedder.embed_chunks(chunks)

# 3. 保存 embeddings + chunks + meta
np.save(os.path.join(data_dir, "embeddings.npy"), embeddings)
with open(os.path.join(data_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
with open(os.path.join(data_dir, "meta_chunks.jsonl"), "w", encoding="utf-8") as f:
    for m in metas:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# 4. 建索引
index = build_faiss_index(embeddings)
faiss.write_index(index, os.path.join(data_dir, "faiss.index"))
print("FAISS index built!")