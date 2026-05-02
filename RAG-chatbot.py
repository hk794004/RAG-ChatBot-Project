# Import Libraries_______________________________________________

import os
import streamlit as st
import dotenv 
import tempfile
import chromadb

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

st.set_page_config(page_title="ChatBot Agent",layout = "wide")
st.title("🤖 RAG ChatBot Agent Read Mutiple PDF Document + Chat History ")
st.caption("Read PDF Document---> Ask Question--->Get Ansawers")

st.divider()

with st.sidebar:

    st.header("⚙️ Control")

    api_input = st.text_input(
        "GROQ_API_KEY",
        type="password",
    )

Key = api_input if api_input else GROQ_API_KEY

if not Key:
    st.error("API KEY Missing")
    st.stop()


# Create Embeddings___________________________

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs = {"normalize_embeddings" : True},
    )

LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=Key,
)

# Create File Uploader__________________________________

file_uploader = st.file_uploader(
    "Upload PDFs",
    type="pdf",
     accept_multiple_files=True,
)

# Upload PDFs___________________________________________

if not file_uploader:
    st.warning("⚠️ Warning Please Upload PDFs Document")
    st.stop()

all_docs = []
tmp_path = []

for csv in file_uploader:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") # Temporary disk pe path bana ra he delete= False matlb disk pe permanent pdf file save krra he 
    temp.write(csv.getvalue()) # pdf file k text ko parh ra he likh ra he disk k andr
    temp.close() # ab close kr raha he 
    tmp_path.append(temp.name) # tmp_path k andr temporary file banayi he disk pe ush ko tmp_path k andr dalra he 

    loader = PyPDFLoader(temp.name) # disk pe jo pdf file bani he temporary path folder bana he ush ka address dera he pypdfloader ko 
    Docs = loader.load() # pdf k file ko read krra he text ko jo andr text mojood he

    for d in Docs:
        d.metadata["source_file"] = csv.name

    all_docs.extend(Docs)

st.success(f"Loaded {len(all_docs)} pages from {len(file_uploader)} PDFs")

# Clean Path___________________________________________________________________________________________

for clean in tmp_path:
    try:
        os.remove(clean)
    except Exception as e:
        pass

# Chunking Split Text___________________________________________________________________

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 100,
    )

Split = text_splitter.split_documents(all_docs)

# VectorStore_____________________________________________________________________________

INDEX_IDR = "chroma_index"

vectorstore = Chroma.from_documents(
        Split,
        embeddings,
        client=chromadb.Client(),
    )

retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k" : 5, "fetch_k" : 20}
    )

st.sidebar.write(f"🔍 Indexed {len(Split)} chunks for retriveal")

# Helper : format docs for stuffing ________________________

def _join_docs(docs, max_chars=7000):
    chunk, total = [], 0

    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunk.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunk)

# Prompt___________________________________________________

contextualize_q_prompt = ChatPromptTemplate.from_messages([

    ("system",
    "Rewrite the user latest question into a standalone search query using the chat history for the context."
    "Return only the rewritten query, no extra text."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
    
    ])

qa_prompt = ChatPromptTemplate.from_messages([

    ("system",
    "You are a Strict RAG assistant. You must ansawer using ONLY the provided context.\n"
    "If the context does not Contain the ansawer, reply exactly:\n"
    "Out of Scope--not found in provided document.\n"
    "Do NOT use outside knowledge. \n\n"
    "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")

    ])

# Session State for chat history (multi sessions)_________________________

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

chat_hist = st.session_state["chat_history"]

def get_history(session_id):
    if session_id not in chat_hist:
        chat_hist[session_id] = ChatMessageHistory()
    return chat_hist[session_id]
    
# Chat UI Input __________________________________________________________

session_id = st.text_input("🆔 Session_ID", value="default")
user_q = st.chat_input("💬 Ask a Question ...")

# Session_State for chat history here_____________________________________

if user_q:
    history = get_history(session_id)

# Rewrite Question with history___________________________________________

    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        )

    standalone_q = LLM.invoke(rewrite_msgs).content.strip()

    # Retrieve Chunks_____________________________________________________

    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of Scope -- not found in provided documents."
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        st.stop()

    # Build Context Strings________________________________________________

    context_str = _join_docs(docs)

    # Asking final Question with stuffed context___________________________

    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str,
        )

    answer = LLM.invoke(qa_msgs).content

    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # Debug Panels_______________________________________

    with st.expander("Debug : Rewritten Query & Retrieval"):
        st.write("** Rewritten (standalone) query : **")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"** {i}. {doc.metadata.get('source_file','Unknown')} (p {doc.metadata.get('page','?')}) **")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

##########################################################################################################
