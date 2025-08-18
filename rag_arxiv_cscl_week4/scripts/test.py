import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from rag.pdf_utils import extract_text_from_pdf

if __name__ == "__main__":
    text = extract_text_from_pdf("data/pdfs/2508.10123v1.pdf")
    print(text[:500])  # 打印前 500 个字符

from rag.text_utils import chunk_text
chunks =  chunk_text(text)
print(len(chunks), chunks[0][:100])

from rag.embed_utils import Embedder
embedder = Embedder()
vecs = embedder.embed_chunks(chunks[:5])
print(vecs.shape) # (5, 384) 

import numpy as np
from rag.index_utils import build_faiss_index, search_faiss

index = build_faiss_index(vecs)
D, I = search_faiss(index, vecs[:1], k=3)
print(I, D)

