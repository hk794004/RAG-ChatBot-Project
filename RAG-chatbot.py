# Import Libraries_______________________________________________

import os
import streamlit as st
import dotenv 
import tempfile
import hashlib

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma 

# Load API_______________________________________________

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit Page Setup_______________________________________________

st.set_page_config(page_title="ChatBot Agent", layout="wide")
st.title("🤖 RAG ChatBot Agent Read Multiple PDF Document + Chat History")
st.caption("Read PDF Document → Ask Question → Get Answers")

st.divider()

with st.sidebar:
    st.header("⚙️ Control")

    api_input = st.text_input("GROQ_API_KEY", type="password")

Key = api_input if api_input else GROQ_API_KEY

if not Key:
    st.error("API KEY Missing")
    st.stop()

# Embeddings & LLM_______________________________________________

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=Key,
    )

embeddings = get_embeddings()
LLM = get_llm()

# 🔥 Hash function (duplicate control)
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# 🔥 Persistent Vector Store
@st.cache_resource
def get_vectorstore():
    return Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

vectorstore = get_vectorstore()

# File Uploader__________________________________

file_uploader = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
)

if not file_uploader:
    st.warning("⚠️ Please Upload PDFs Document")
    st.stop()

# Load PDFs___________________________________________

all_docs = []
tmp_path = []

for pdf in file_uploader:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp.write(pdf.getvalue())
    temp.close()

    tmp_path.append(temp.name)

    loader = PyPDFLoader(temp.name)
    Docs = loader.load()

    for d in Docs:
        d.metadata["source_file"] = pdf.name

    all_docs.extend(Docs)

st.success(f"Loaded {len(all_docs)} pages from {len(file_uploader)} PDFs")

# Clean temp files
for path in tmp_path:
    try:
        os.unlink(path)
    except:
        pass

# Chunking___________________________________________

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
)

Split = text_splitter.split_documents(all_docs)

# 🔥 Existing IDs (to avoid duplicates)
existing_ids = set()
try:
    data = vectorstore.get(include=["metadatas"])
    for m in data["metadatas"]:
        if m and "chunk_id" in m:
            existing_ids.add(m["chunk_id"])
except:
    pass

# 🔥 Filter new chunks only
new_docs = []
new_ids = []

for doc in Split:
    chunk_id = get_hash(doc.page_content)

    if chunk_id not in existing_ids:
        doc.metadata["chunk_id"] = chunk_id
        new_docs.append(doc)
        new_ids.append(chunk_id)

# 🔥 Add only new chunks
if new_docs:
    vectorstore.add_documents(new_docs, ids=new_ids)
    st.success(f"✅ Added {len(new_docs)} new chunks")
else:
    st.info("ℹ️ No new content (already indexed)")

# Retriever___________________________________________

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.sidebar.write(f"🔍 Total chunks in DB: {len(vectorstore.get()['ids'])}")

# Helper___________________________________________

def _join_docs(docs, max_chars=7000):
    chunk, total = [], 0

    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunk.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunk)

# Prompts___________________________________________

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user latest question into a standalone search query using chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You must answer ONLY from context.\n"
     "If not found say: Out of Scope -- not found in provided document.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Chat History___________________________________________

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

def get_history(session_id):
    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]

# Chat UI___________________________________________

session_id = st.text_input("🆔 Session_ID", value="default")
user_q = st.chat_input("💬 Ask a Question ...")

if user_q:
    history = get_history(session_id)

    # 🔹 Rewrite question
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
    )

    standalone_q = LLM.invoke(rewrite_msgs).content.strip()

    # 🔹 Retrieve
    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of Scope -- not found in provided documents."
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)

        history.add_user_message(user_q)
        history.add_ai_message(answer)

        # Debug بھی empty case میں دکھاؤ
        with st.expander("🔍 Debug: Query & Retrieval"):
            st.write("**Standalone Query:**")
            st.code(standalone_q or "(empty)")

            st.write("**Retrieved Chunks:** 0")

        st.stop()

    # 🔹 Context build
    
    context_str = _join_docs(docs)

    # 🔹 Final answer

    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str,
    )

    answer = LLM.invoke(qa_msgs).content

    # 🔹 Display chat

    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # ===============================
    # 🔍 DEBUG EXPANDER (IMPORTANT)
    # ===============================

    with st.expander("🔍 Debug: Rewritten Query & Retrieval"):

        st.write("Standalone Query")
        st.code(standalone_q or "(empty)", language="text")

        st.write(f"### 📊 Retrieved {len(docs)} chunk(s)")

    # ===============================
    # 📄 RETRIEVED CHUNKS EXPANDER
    # ===============================

    with st.expander("📄 Retrieved Chunks (Top Results)"):

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")

            st.markdown(f"**{i}. {source} (Page {page})**")
            st.write(
                doc.page_content[:500] +
                ("..." if len(doc.page_content) > 500 else "")
            )
            st.divider()
