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
        filenames = [new_file_path]
    else:
        filenames = [os.path.join("docs", f) for f in os.listdir("docs")]

    for filepath in filenames:
        filename = os.path.basename(filepath)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath)
            try:
                documents.extend(loader.load())
                print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        elif filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(filepath)
            try:
                documents.extend(loader.load())
                print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
            try:
                documents.extend(loader.load())
                print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        elif filename.endswith(".csv"):
            try:
                documents.extend(load_csv_as_documents(filepath))
                print(f"✅ Loaded (custom CSV): {filename}")
            except Exception as e:
                print(f"❌ Failed to load CSV {filename}: {e}")
        else:
            print(f"⏩ Skipping unsupported file: {filename}")
            continue

    if not documents:
        raise ValueError("No documents loaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    if incremental and os.path.exists("vector_index/index.faiss"):
        print("➕ Adding new chunks to existing index")
        db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        print("🆕 Creating new FAISS index from scratch")
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vector_index")
    print("✅ Vector index saved")
    return db
