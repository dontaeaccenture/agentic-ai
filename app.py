import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from rag_setup import create_faiss_index  # 🔁 FAISS index regeneration

load_dotenv()

# ✅ Azure OpenAI client setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = Flask(__name__)

# ✅ Embedding setup
embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# 🧠 Rebuild FAISS vector index if missing
VECTOR_PATH = "vector_index"
if not os.path.exists(f"{VECTOR_PATH}/index.faiss"):
    print("🔄 No vector index found. Rebuilding...")
    db = create_faiss_index()
else:
    db = FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agent", methods=["POST"])
def agent_chat():
    user_input = request.json.get("message", "")

    # 🔍 Retrieve relevant chunks from embedded documents
    docs = db.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        {
            "role": "system",
            "content": f"""
You are a senior cloud architect tasked with creating a structured, realistic 3-year migration plan to AWS and Azure.
Use only the data below as your source of truth:
\n\n{context}\n\n
Structure your answer by year (Year 1, Year 2, Year 3), and include phases like: discovery, planning, POC, lift-and-shift, optimization, and cloud-native rebuild.
Assume the user wants performance, security, and cost-efficiency.
"""
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # 🧠 Query Azure OpenAI (GPT-4o)
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    reply = response.choices[0].message.content
    return jsonify({"reply": reply})

@app.route("/docs")
def get_docs():
    docs = db.similarity_search("example", k=20)
    previews = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content[:500]
        }
        for doc in docs
    ]
    return jsonify({"documents": previews})

if __name__ == "__main__":
    app.run(debug=True)
