import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray):

    # 创建 FAISS 索引并加入所有向量
    # embeddings: numpy array, shape (n_chunks, dim)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) #内积索引，需embeddings 归一化
    index.add(embeddings)
    return index

def search_faiss(index, query_vector, k=3):
   
    # 在索引中检索最相似的 k 个结果
    # shape (n_queries, dim)
    #返回： (D, I)，分别是距离和索引
    
    D, I = index.search(query_vector, k)
    return D, I