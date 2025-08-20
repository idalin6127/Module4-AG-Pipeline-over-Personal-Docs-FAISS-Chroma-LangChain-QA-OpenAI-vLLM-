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

# 8. `ssh -L 8000:localhost:8000 jovyan@216.81.245.8 -p 30932`

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


### API Version Embedding

# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os


load_dotenv()  # This will load variables from the .env file into the environment

# Now you can access the API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API Key not found. Please add OPENAI_API_KEY to your .env file.")

# 1) LOAD your resume & docs
resume = PyPDFLoader("C:\\AI_Coop\\Homework\\Week4\\class4\\AI_Engineer_Ida_Lin_resume.pdf").load()
extras = TextLoader("C:\\AI_Coop\\Homework\\Week4\\class4\\portfolio_notes_Ida_Lin.txt").load()
docs = resume + extras

# 2) Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3) Create embeddings and vector store
emb = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, emb)

# 4) MAKE RAG chain
llm = OpenAI(temperature=0, openai_api_key=api_key)
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k":3})
)

# 5) Function to ask questions
def ask_me(question):
    print(agent.run(question))

# Example usage
if __name__=="__main__":
    ask_me("What is Ida's AI experience?")


