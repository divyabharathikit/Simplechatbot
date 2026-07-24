import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Insurance RAG Chatbot")

st.title("🤖 Insurance RAG Chatbot")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(
    model="llama3.2"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask your insurance question")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    docs = db.similarity_search(
        question,
        k=3
    )

    context = ""

    for doc in docs:
        context += doc.page_content + "\n"

    prompt = f"""
Use only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)