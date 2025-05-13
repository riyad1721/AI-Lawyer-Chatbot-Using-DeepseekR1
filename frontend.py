from rag_pipeline import answer_query, retrieve_docs, llm_model
import streamlit as st
from vector_database import upload_pdf, load_pdf, create_chunks, get_embedding_model, create_vector_store

st.set_page_config(page_title="AI Lawyer", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            font-size: 18px;
            padding: 0.5em 1.5em;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

ollama_model_name = "deepseek-r1:7b"
FAISS_DB_PATH = "vectorstore/db_faiss"
pdfs_directory = './pdfs_data/'

# --- Header ---
# st.image("https://i.imgur.com/dR3zGBj.png", use_column_width=True)  # You can replace this with a local file or another URL
st.title("⚖️ AI Lawyer Assistant")
st.subheader("Ask legal questions based on your uploaded documents")

# --- Upload + Query UI ---
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

with col2:
    user_query = st.text_area("💬 Ask Your Question", height=150, placeholder="e.g., What does Article 19 of this document say?")

ask_question = st.button("Ask AI Lawyer")

# --- Processing Section ---
if ask_question:
    if uploaded_file and user_query:
        with st.spinner("Processing document and generating answer..."):
            upload_pdf(uploaded_file)
            documents = load_pdf(pdfs_directory + uploaded_file.name)
            text_chunks = create_chunks(documents)
            faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks, ollama_model_name)
            retrieved_docs = retrieve_docs(faiss_db, user_query)
            response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(response.content)

    else:
        st.error("❗ Please upload a valid PDF file and ask a question.")