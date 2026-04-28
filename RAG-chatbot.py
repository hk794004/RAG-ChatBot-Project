import streamlit as st
import os
import dotenv
import tempfile

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API KEY

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Streamlit page setup

st.set_page_config(page_title="RAG Chatot", layout="wide")
st.title("RAG ChatBot Agent Q&A With Multiples PDF Pages + Chat History")
st.caption("Build With Streamlit + GROQ API Cloud + HuggingFace + LLM")

# Sidebar

with st.sidebar:

    st.header("⚙️ Config")

    API_Inpt = st.text_input(
        "GROQ API KEY",
        type="password",
    )

API_KEY = API_Inpt if API_Inpt else GROQ_API_KEY

if not API_KEY:
    st.warning("API KEY Missing Please Insert API KEY into Sidebar")
    st.stop()
else:
    st.sidebar.success("API KEY Loaded")

with st.sidebar:
    st.caption("Upload PDF --> Ask Question --> Get Ansawers")

# Create Embeddings OR LLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

LLM = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=API_KEY,
)

Upload_Files = st.file_uploader(
    "Upload PDF Files",
    type="PDF",
    accept_multiple_files=True
)

if not Upload_Files:
    st.info("Please Upload One Or More PDF to Start")
    st.stop()

Doc = []
Path = []

for pdf in Upload_Files:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp.write(pdf.getvalue())
    temp.close()
    Path.append(temp.name)

    loader = PyPDFLoader(temp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name

    Doc.extend(docs)

st.success(f" Loaded {len(Doc)} Pages From {len(Upload_Files)}")

# Clean Temp File

for clean in Path:
    try:
        os.unlink(clean)
    except Exception as e:
        pass

# Chunks (Split Text)

text_split = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120,
)

Split = text_split.split_documents(Doc)

# Vector Store

INDEX_DIR = "chroma_index"

Vectorstore = Chroma.from_documents(
    Split,
    embeddings,
    persist_directory=INDEX_DIR,
)

Retriver = Vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.sidebar.write(f" Indexed {len(Split)} Chunks For Retrival")

# Helper Document For Stuffing

def join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's latest question into a standalone search query using the chat history for context. Return only the rewritten query, no extra text."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a STRICT RAG assistant. You must answer using ONLY the provided context.\n"
        "If the context does NOT contain the answer, reply exactly:\n"
        "'Out of scope - not found in provided documents.'\n"
        "DO NOT use outside knowledge.\n\n"
        "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = {}

chat_history = st.session_state['chat_history']

Session_ID = "default"

def get_history(session_id):
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]

Session_ID = st.text_input("👤 Session_ID", value='default')

User_Q = st.chat_input("💬 Ask a Question...")

# Session State For Chat History Here

if User_Q:
    history = get_history(Session_ID)

    # Rewrite Question With History

    rewrite_msgs = prompt.format_messages(
        chat_history=history.messages,
        input=User_Q,
    )

    standalone_q = LLM.invoke(rewrite_msgs).content.strip()

    # Retrieve Chunks

    docs = Retriver.invoke(standalone_q)

    if not docs:
        answer = "Out of scope - not found in provided documents."
        st.chat_message("user").write(User_Q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(User_Q)
        history.add_ai_message(answer)
        st.stop()

    # 3) Build context string

    context_str = join_docs(docs)

    # Asking final question with stuffed context

    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=User_Q,
        context=context_str
    )
    
    answer = LLM.invoke(qa_msgs).content

    st.chat_message("user").write(User_Q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(User_Q)
    history.add_ai_message(answer)

    # Debug panels

    with st.expander("🔍 Debug: Rewritten Query & Retrieval"):
        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📄 Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):

            st.markdown(f"**{i}. {doc.metadata.get('source_file', 'Unknown')} (p{doc.metadata.get('page', '?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
