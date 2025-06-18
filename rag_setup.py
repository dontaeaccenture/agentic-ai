import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader
)
from langchain.schema import Document

load_dotenv()

def load_csv_as_documents(filepath):
    docs = []
    df = pd.read_csv(filepath)

    for _, row in df.iterrows():
        content = ", ".join(f"{col}: {val}" for col, val in row.items())
        doc = Document(page_content=content, metadata={"source": os.path.basename(filepath)})
        docs.append(doc)

    return docs

def create_faiss_index(incremental=False, new_file_path=None):
    documents = []

    if incremental and new_file_path:
        # Handle string or list of paths
        if isinstance(new_file_path, str):
            filenames = [new_file_path]
        else:
            filenames = new_file_path
    else:
        filenames = [os.path.join("docs", f) for f in os.listdir("docs")]

    for filepath in filenames:
        filename = os.path.basename(filepath)
        ext = filename.lower().rsplit('.', 1)[-1]

        try:
            if ext == "txt":
                loader = TextLoader(filepath)
                documents.extend(loader.load())
            elif ext == "pdf":
                loader = UnstructuredPDFLoader(filepath)
                documents.extend(loader.load())
            elif ext == "docx":
                loader = UnstructuredWordDocumentLoader(filepath)
                documents.extend(loader.load())
            elif ext == "csv":
                documents.extend(load_csv_as_documents(filepath))
            else:
                print(f"⏩ Skipping unsupported file: {filename}")
                continue

            print(f"✅ Loaded: {filename}")

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    if not documents:
        raise ValueError("No documents loaded for indexing.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings setup
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    index_path = "vector_index"

    if incremental and os.path.exists(f"{index_path}/index.faiss"):
        print("➕ Appending to existing FAISS index")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        print("🆕 Creating new FAISS index")
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(index_path)
    print("✅ FAISS index saved to disk")
    return db
