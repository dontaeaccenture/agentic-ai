import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
)

load_dotenv()

def create_faiss_index():
    docs_folder = "docs"
    documents = []

    for filename in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.endswith(".csv"):
            loader = CSVLoader(filepath)
        else:
            print(f"⏩ Skipping unsupported file: {filename}")
            continue

        try:
            documents.extend(loader.load())
            print(f"✅ Loaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    if not documents:
        raise ValueError("No documents loaded. Ensure docs/ contains valid files.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vector_index")

    print("✅ All documents processed and vector index saved.")
    return db
