from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

pdfs_directory = './pdfs_data/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

file_path = 'human-declaration-of-human-rights.pdf'
full_file_path = os.path.join(pdfs_directory, file_path)
documents = load_pdf(full_file_path)
# print("PDF pages: ",len(documents))

text_chunks = create_chunks(documents)
# print("Chunks count: ", len(text_chunks))


ollama_model_name="deepseek-r1:7b"
def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

# FAISS_DB_PATH="vectorstore/db_faiss"
# faiss_db=FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
# faiss_db.save_local(FAISS_DB_PATH)

def create_vector_store(db_path, chunks, model_name):
    faiss_db = FAISS.from_documents(chunks, get_embedding_model(model_name))
    faiss_db.save_local(db_path)
    return faiss_db