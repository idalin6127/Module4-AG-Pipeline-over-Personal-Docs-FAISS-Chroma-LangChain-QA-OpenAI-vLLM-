import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pdf_utils import extract_text_from_pdf
from rag.text_utils import chunk_text
from rag.embed_utils import Embedder
import numpy as np
from rag.index_utils import build_faiss_index, search_faiss

if __name__ == "__main__":
    # Create output file
    with open("notebooks/test_results.txt", "w", encoding="utf-8") as f:
        # Test PDF text extraction
        text = extract_text_from_pdf("data/pdfs/2508.10123v1.pdf")
        f.write("Extracted text (first 500 chars):\n")
        f.write(text[:500])
        f.write("\n\n" + "="*50 + "\n\n")
        
        # Test text chunking
        chunks = chunk_text(text)
        f.write(f"Number of chunks: {len(chunks)}\n")
        f.write(f"First chunk (first 100 chars): {chunks[0][:100]}\n")
        f.write("\n" + "="*50 + "\n\n")
        
        # Test embedding
        embedder = Embedder()
        vecs = embedder.embed_chunks(chunks[:5])
        f.write(f"Embedding shape: {vecs.shape}\n")  # Should be (5, 384)
        f.write("\n" + "="*50 + "\n\n")
        
        # Test FAISS index building and searching
        index = build_faiss_index(vecs)
        D, I = search_faiss(index, vecs[:1], k=3)
        f.write(f"FAISS Search Results:\n")
        f.write(f"Indices (I): {I}\n")
        f.write(f"Distances (D): {D}\n")
        
        # Also print to console
        print("Results saved to notebooks/test_results.txt")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Embedding shape: {vecs.shape}")
        print(f"FAISS indices: {I}")
        print(f"FAISS distances: {D}")
