import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import AzureOpenAI
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from rag_setup import create_faiss_index # 👈 Make sure load_and_split_file exists
from planner_agent import plan_and_execute
from models import SessionLocal, ChatHistory, MigrationStats  # Already imported ChatHistory

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

VECTOR_PATH = "vector_index"
UPLOAD_FOLDER = "docs"
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 🧠 Load or build FAISS vector index
if not os.path.exists(f"{VECTOR_PATH}/index.faiss"):
    print("🔄 No vector index found. Rebuilding...")
    db = create_faiss_index()
else:
    db = FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)

def update_migration_stat(field_name, increment=1):
    db_session = SessionLocal()
    stats = db_session.query(MigrationStats).first()
    
    if not stats:
        stats = MigrationStats()
        setattr(stats, field_name, increment)
        db_session.add(stats)
    else:
        current = getattr(stats, field_name, 0) or 0
        setattr(stats, field_name, current + increment)

    db_session.commit()
    db_session.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agent", methods=["POST"])
def agent_chat():
    user_input = request.json.get("message", "")
    docs = db.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    keywords = ["migration", "cloud", "aws", "azure", "infrastructure", "plan", "roadmap", "rehost", "replatform"]
    is_migration_request = any(word in user_input.lower() for word in keywords)

    if is_migration_request:
        system_prompt = f"""
You are a senior cloud architect tasked with creating a structured, realistic 3-year migration plan to AWS and Azure.
Use only the data below as your source of truth:
\n\n{context}\n\n
Structure your answer by year (Year 1, Year 2, Year 3), and include phases like: discovery, planning, POC, lift-and-shift, optimization, and cloud-native rebuild.
Assume the user wants performance, security, and cost-efficiency.
"""
        messages = [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_input }
        ]
        update_migration_stat("plans_generated")


    else:
        messages = [
        {
            "role": "system",
            "content": f"""
You are a senior cloud architect tasked with creating a structured, realistic 3-year migration plan to AWS and Azure.
Use only the data below as your source of truth:
\n\n{context}
"""
        },
        { "role": "user", "content": user_input }
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    reply = response.choices[0].message.content

    # ✅ Make sure DB code is INSIDE this function
    # ✅ Save chat + increment metrics
    try:
        db_session = SessionLocal()
        # Save chat
        chat_entry = ChatHistory(
            session_id=request.remote_addr,
            user_input=user_input,
            ai_response=reply
        )
        db_session.add(chat_entry)

        update_migration_stat("user_messages")

    except Exception as e:
        print(f"⚠️ Failed to save chat or update stats: {e}")

    return jsonify({"reply": reply})


@app.route("/agent/vanilla", methods=["POST"])
def agent_vanilla():
    user_input = request.json.get("message", "")
    docs = db.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        {
            "role": "system",
            "content": f"""
You are a senior cloud architect tasked with creating a structured, realistic 3-year migration plan to AWS and Azure.
Use only the data below as your source of truth:
\n\n{context}
"""
        },
        { "role": "user", "content": user_input }
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    reply = response.choices[0].message.content
    # ✅ Make sure DB code is INSIDE this function
    try:
        db_session = SessionLocal()
        chat_entry = ChatHistory(
            session_id=request.remote_addr,
            user_input=user_input,
            ai_response=reply
        )
        db_session.add(chat_entry)
        db_session.commit()
        db_session.close()
    except Exception as e:
        print(f"⚠️ Failed to save chat to DB: {e}")

    return jsonify({"reply": reply})
@app.route("/history", methods=["GET"])
def get_history():
    try:
        db_session = SessionLocal()
        records = db_session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).limit(50).all()
        db_session.close()

        history = [
            {
                "id": chat.id,
                "session_id": chat.session_id,
                "user_input": chat.user_input,
                "ai_response": chat.ai_response,
                "timestamp": chat.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            for chat in records
        ]

        return jsonify({"history": history})

    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return jsonify({"error": "Failed to fetch chat history."}), 500

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

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            global db
            db = create_faiss_index(incremental=True, new_file_path=file_path)

            # ✅ Increment uploaded document count
            db_session = SessionLocal()
            stats = db_session.query(MigrationStats).first()
            if not stats:
                stats = MigrationStats()
                db_session.add(stats)
                stats.documents_uploaded = 1
            else:
                stats.documents_uploaded = (stats.documents_uploaded or 0) + 1

            db_session.commit()
            db_session.close()

            return jsonify({
                "status": "success",
                "message": f"✅ File '{filename}' uploaded and indexed."
            }), 200

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"❌ Error updating index: {str(e)}"
            }), 500

    return jsonify({
        "status": "error",
        "message": "Unsupported file type"
    }), 400

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        db_session = SessionLocal()
        stats = db_session.query(MigrationStats).first()
        db_session.close()

        if not stats:
            return jsonify({
                "plans_generated": 0,
                "documents_uploaded": 0,
                "user_messages": 0
            })

        return jsonify({
            "plans_generated": stats.plans_generated,
            "documents_uploaded": stats.documents_uploaded,
            "user_messages": stats.user_messages
        })
    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")
        return jsonify({"error": "Could not retrieve metrics"}), 500

if __name__ == "__main__":
    app.run(debug=True)
