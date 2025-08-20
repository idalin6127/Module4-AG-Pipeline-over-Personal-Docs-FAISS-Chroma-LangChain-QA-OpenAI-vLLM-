# Class Project: “My Resume AI Assistant”
## Project Vision
# - Ingest: Take your résumé (PDF or DOCX) plus any supporting portfolio documents.

# - RAG Layer (Week 4): Build a vector index over those documents so the agent can retrieve exact snippets (experience bullet, project detail) to answer user questions like “What technologies did I use on Project X?”

# - SFT Layer (Weeks 5–7+): Fine‑tune a small LLM on your own Q&A pairs or conversational logs so it internalizes style, tone, and deeper context—making it feel like you speaking, not just quoting text.

# ## Inference AI set up
# ### 1. GPU Set Up
# sign in the console at: https://console.inference.ai/

# 1. `Create New Server` 
# 2. Default setting, `Configure`
# 3. Advanced Options -> Enabled the `SSH Access`
# 4. `Manage` SSH Access 
#     - open your terminal type: ` cat ~/.ssh/id_rsa.pub `

#     - If the .ssh directory doesn't exist or you don't find any public key files, you'll need to generate a new SSH key pair using the ssh-keygen command. check [this](https://docs.github.com/en/enterprise-cloud@latest/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys)
    
#     - Copy the ssh public key and paste into inference AI ssh key management box.
    
#     - `close`

# 5. `Deploy Server`

# 6. In the Running Instances, click `Show Details`

# 7. copy SSH Access such that `ssh jovyan@216.81.245.8 -p 30932`

# 8. `ssh -L 8000:localhost:8000 jovyan@216.81.245.8 -p 30932`(run it in local cmd)

# 9. pip install vllm (virtual server)

# 10. pip3 install --user huggingface_hub (virtual server)

# export PATH=$PATH:~/.local/bin
# echo $PATH
# huggingface-cli login
# huggingface-cli whoami


# ### 2. System Set Up (in the same remote console)

# 1. Download the Vllm using `pip install vllm` 
# - [What is vllm ?](https://docs.vllm.ai/en/latest/)
#     - vLLM is an open-source library designed for efficient and high-throughput serving of Large Language Models (LLMs). It focuses on optimizing memory management during inference, particularly with techniques like "PagedAttention," which reduces KV-cache waste. vLLM offers faster serving speeds, seamless integration with popular Hugging Face models, and supports various decoding algorithms like parallel sampling and beam search. 

# 2. Set up HuggingFace
# - go to https://huggingface.co/settings/tokens to generate the new token
# - `huggingface-cli login`

# 3. Download model from HuggingFace 
#     ```
#     python3 -m vllm.entrypoints.openai.api_server \
#     --model HuggingFaceH4/zephyr-7b-alpha \
#     --port 8000 \
#     --gpu-memory-utilization 0.99 \
#     --max-model-len 2048
#     ```


## GPU Version Embedding
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Load your documents
resume = PyPDFLoader("C:\\AI_Coop\\Homework\\Week4\\class4\\AI_Engineer_Ida_Lin_resume.pdf").load()
extras = TextLoader("C:\\AI_Coop\\Homework\\Week4\\class4\\portfolio_notes_Ida_Lin.txt").load()
docs = resume + extras

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create embeddings with OpenAI (you can replace this later with local embedding models)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embedding_model)



# 4. Connect to your vLLM server running on port 8000 (via SSH tunnel or locally)
llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="not-needed",
    model_name="HuggingFaceH4/zephyr-7b-alpha"  # matches your vLLM launch
)

# 5. Retrieval QA chain
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 6. Ask a question
def ask_me(question):
    print(agent.run(question))

if __name__ == "__main__":
    ask_me("What is Ida's AI experience?")
