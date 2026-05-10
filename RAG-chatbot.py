# ================= IMPORTS =================
import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ================= LOAD ENV =================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# ================= STREAMLIT UI =================
st.set_page_config(page_title="RAG ChatBot", layout="wide")
st.title("🤖 RAG ChatBot (Fixed Version)")
st.caption("PDF → Chat → Answers")

# ================= API KEY =================
with st.sidebar:
    api_input = st.text_input("GROQ_API_KEY", type="password")

Key = api_input if api_input else API_KEY

if not Key:
    st.error("Missing API Key")
    st.stop()

st.sidebar.success("API Key Loaded")

# ================= EMBEDDINGS =================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ================= LLM =================
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=Key,
    )

# ================= FILE UPLOAD =================
with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

if not uploaded_files:
    st.warning("Upload PDFs")
    st.stop()

# ================= LOAD PDFS =================
all_docs = []
tmp_files = []

for pdf in uploaded_files:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_files.append(tmp.name)

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name

    all_docs.extend(docs)

# cleanup temp files
for f in tmp_files:
    try:
        os.unlink(f)
    except:
        pass

st.sidebar.success(f"Loaded {len(all_docs)} pages")

# ================= CHUNKING =================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
)

chunks = splitter.split_documents(all_docs)

# ================= VECTORSTORE (FIXED) =================

# detect file change
file_hash = hash(str([f.name for f in uploaded_files]))

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.session_state.file_hash != file_hash:

    st.session_state.file_hash = file_hash

    st.session_state.vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20},
)

st.sidebar.write(f"🔍 Indexed {len(chunks)} chunks")

# ================= CHAT MEMORY =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

def get_history(session_id):
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

# ================= PROMPTS =================
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite question into standalone query using chat history. "
     "Return only query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. Use ONLY context.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ================= HELP FUNCTION =================
def _join_docs(docs, max_chars=7000):
    out, total = [], 0
    for d in docs:
        if total + len(d.page_content) > max_chars:
            break
        out.append(d.page_content)
        total += len(d.page_content)
    return "\n\n---\n\n".join(out)

# ================= INPUT =================
session_id = st.text_input("Session ID", value="default")
user_q = st.chat_input("Ask question...")

# ================= MAIN FLOW =================
if user_q:

    history = get_history(session_id)
    llm = get_llm()

    # rewrite question
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
    )

    standalone_q = llm.invoke(rewrite_msgs).content.strip()

    # retrieve
    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of Scope - not found in documents."
    else:
        context = _join_docs(docs)

        qa_msgs = qa_prompt.format_messages(
            chat_history=history.messages,
            input=user_q,
            context=context,
        )

        answer = llm.invoke(qa_msgs).content

    # chat output
    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # debug
    with st.expander("Debug"):
        st.code(standalone_q)
        st.write(len(docs), "chunks retrieved")
