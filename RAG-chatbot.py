# Import Libraries_______________________________________________

import os
import streamlit as st
import dotenv 
import tempfile

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    st.error("API KEY is missing")
    st.stop()
else:
    st.sidebar.success("Api Key Loaded")


# Create Embeddings___________________________

@st.cache_resource
def get_embeddings():
    
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings" : True},
    )
    return embeddings

@st.cache_resource
def Get_LLM():
    
    LLM = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=Key,
    streaming=False
    )
    return LLM

# Create File Uploader__________________________________

file_uploader = st.sidebar.file_uploader(
    "Upload Word Doc",
    type="pdf",
     accept_multiple_files=True,
)

# Upload PDFs___________________________________________

if not file_uploader:
    st.warning("⚠️ Warning Please Upload PDF Document")
    st.stop()

all_docs = []
tmp_path = []

for csv in file_uploader:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") 
    temp.write(csv.getvalue())
    temp.close() 
    tmp_path.append(temp.name) 

    loader = PyPDFLoader(temp.name) 
    Docs = loader.load() 

    for d in Docs:
        d.metadata["source_file"] = csv.name

    all_docs.extend(Docs)

st.sidebar.success(f"Loaded {len(all_docs)} pages from {len(file_uploader)} Word Document")

# Clean Path___________________________________________________________________________________________

for clean in tmp_path:
    try:
        os.unlink(clean)
    except Exception as e:
        pass

# Chunking Split Text___________________________________________________________________

@st.cache_resource
def chunks():

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1200,
            chunk_overlap = 100,
    )

    return text_splitter

text = chunks()

Split = text.split_documents(all_docs)

# VectorStore_____________________________________________________________________________

current_files = sorted([f.name for f in file_uploader])

if (
    "vectorstore" not in st.session_state or
    st.session_state.get("loaded_files") != current_files
):
    st.session_state["vectorstore"] = FAISS.from_documents(
        Split,
        get_embeddings(),
    )
    st.session_state["loaded_files"] = current_files
    st.session_state.pop("chat_history", None)


retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
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
     "You are an expert assistant that rewrites user questions into standalone questions for document retrieval.\n"
     "Your job is to use the chat history ONLY if needed to understand context.\n\n"
     
     "Rules:\n"
     "- Convert the user's question into a clear, self-contained question\n"
     "- Do NOT answer the question\n"
     "- Do NOT add explanations\n"
     "- Do NOT assume missing facts not in chat history\n"
     "- Keep it short and precise\n"
     "- The output must be only the rewritten question"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert AI assistant specialized in analyzing documents.\n\n"

     "Your role:\n"
     "- Answer ONLY using the given context from the document\n"
     "- Do NOT use outside knowledge\n"
     "- If the answer is NOT in the context, do NOT say 'not available'\n"
     "  Instead: Suggest 2-3 related topics or stories that ARE present in the document\n"
     "  Format:\n"
     "  'This topic is not in the document, but here are related things I found:\n"
     "   • [suggestion 1]\n"
     "   • [suggestion 2]'\n\n"

     "- Never hallucinate\n"
     "- Always stay within document content only"
    ),

    ("human",
     "Context from document:\n{context}\n\n"
     "Question:\n{input}")
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

# ✅ Pehle poori history dikhao — har rerun pe
history = get_history(session_id)

for msg in history.messages:
    role = getattr(msg, "type", "")

    if role == "human":
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("ai").write(msg.content)

User_Input = st.chat_input("💬 Ask a Question ...")

if User_Input:

    st.chat_message("human").write(User_Input)


    # Rewrite Question with history

    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages, 
        input=User_Input,
    )

    LLM = Get_LLM()

    standalone_q = LLM.invoke(rewrite_msgs).content.strip()

    # Retrieve Chunks

    docs = retriever.invoke(standalone_q)


    if not docs:
        answer = "Out of Scope -- not found in provided documents."
        with st.chat_message("assistant"):
            st.write(answer)
        history.add_user_message(User_Input)
        history.add_ai_message(answer)
        st.stop()

    # Build Context

    context_str = _join_docs(docs)

    # Final Answer

    qa_msgs = qa_prompt.format_messages(
        input=User_Input,
        context=context_str,
    )

    answer = LLM.invoke(qa_msgs).content

    with st.chat_message("assistant"):
        st.markdown(answer)

    history.add_user_message(User_Input)
    history.add_ai_message(answer)

    # Debug Panels

    with st.expander("🔍 Debug : Rewritten Query & Retrieval"):

        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📄 Retrieved Chunks"):

        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p {doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
