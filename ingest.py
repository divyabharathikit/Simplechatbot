from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

loader = PyPDFDirectoryLoader("data")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = FAISS.from_documents(chunks, embeddings)

db.save_local("faiss_index")

print("FAISS index created successfully!")