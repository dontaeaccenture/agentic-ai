import os
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory

from io import BytesIO
from dotenv import load_dotenv
from openai import AzureOpenAI
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from rag_setup import create_faiss_index # 👈 Make sure load_and_split_file exists
from planner_agent import plan_and_execute
from models import SessionLocal, ChatHistory, MigrationStats  # Already imported ChatHistory
from sqlalchemy import or_

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

def update_migration_stat(field_name, increment=1, db_session=None):
    internal_session = False

    if db_session is None:
        db_session = SessionLocal()
        internal_session = True

    stats = db_session.query(MigrationStats).first()

    if not stats:
        stats = MigrationStats()
        setattr(stats, field_name, increment)
        db_session.add(stats)
    else:
        current = getattr(stats, field_name, 0) or 0
        setattr(stats, field_name, current + increment)

    if internal_session:
        db_session.commit()
        db_session.close()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agent", methods=["POST"])
def agent_chat():
    user_input = request.json.get("message", "")
    docs = db.similarity_search(user_input, k=6)
    context = "\n\n".join(doc.page_content for doc in docs)

    keywords = ["migration", "cloud", "aws", "azure", "infrastructure", "plan", "roadmap", "rehost", "replatform"]
    is_migration_request = any(word in user_input.lower() for word in keywords)

    if is_migration_request:
        system_prompt = f"""
You are a senior cloud architect tasked with creating a structured, realistic 3-year migration plan to AWS and Azure.

Use only the data below as your source of truth:
\n\n{context}\n\n

Structure your answer by year (Year 1, Year 2, Year 3), and include phases like: discovery, planning, proof of concept (POC), lift-and-shift, optimization, and cloud-native rebuild.

Assume the user wants performance, security, and cost-efficiency.

When recommending AWS or Azure cloud services, include relevant documentation or pricing tools using **Markdown links**. For example:
- [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices)
- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [AWS Pricing Calculator](https://calculator.aws.amazon.com/)
- [AWS Pricing API Docs](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html)

Your response should be professional, actionable, and formatted clearly using Markdown (headings, bullet points, etc.).
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
You are a helpful assistant for cloud migration teams.
Use the following as source context only:
\n\n{context}
"""
            },
            { "role": "user", "content": user_input }
        ]

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.7,
        max_tokens=3000
    )

    reply = response.choices[0].message.content

    try:
        db_session = SessionLocal()
        chat_entry = ChatHistory(
            session_id=request.remote_addr,
            user_input=user_input,
            ai_response=reply
        )
        db_session.add(chat_entry)

        update_migration_stat("user_messages", db_session=db_session)  # reuses active session

        db_session.commit()
        plan_id = chat_entry.id

    except Exception as e:
        print(f"⚠️ Failed to save chat or update stats: {e}")
        plan_id = None

    finally:
        db_session.close()

    return jsonify({
    "reply": reply,
    "plan_id": plan_id if is_migration_request else None
})



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

@app.route("/uploads")
def view_uploaded_documents():
    from langchain_community.vectorstores import FAISS
    docs = db.similarity_search("infrastructure", k=50)
    doc_previews = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content[:500]
        }
        for doc in docs
    ]
    return render_template("uploads.html", documents=doc_previews)

@app.route("/download/<path:filename>")
def download_document(filename):
    return send_from_directory("docs", filename, as_attachment=True)

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        db_session = SessionLocal()
        stats = db_session.query(MigrationStats).first()

        # Dynamically count current valid files in the docs folder
        try:
            files_in_folder = [
                f for f in os.listdir("docs")
                if os.path.isfile(os.path.join("docs", f))
                and f.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS
            ]
            docs_uploaded = len(files_in_folder)
        except FileNotFoundError:
            docs_uploaded = 0

        db_session.close()

        return jsonify({
            "plans_generated": stats.plans_generated if stats else 0,
            "documents_uploaded": docs_uploaded,
            "user_messages": stats.user_messages if stats else 0
        })
    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")
        return jsonify({"error": "Could not retrieve metrics"}), 500


@app.route("/plans/<int:plan_id>", methods=["GET"])
def get_plan(plan_id):
    db_session = SessionLocal()
    plan = db_session.query(ChatHistory).filter(ChatHistory.id == plan_id).first()
    db_session.close()

    if not plan:
        app.logger.warning(f"Plan ID {plan_id} not found.")
        return jsonify({"error": "Plan not found"}), 404

    filename = f"migration_plan_{plan_id}.txt"
    content = f"User Message:\n{plan.user_input}\n\nAI Response:\n{plan.ai_response}"

    # Serve the file as a downloadable response
    return send_file(
        BytesIO(content.encode()),
        as_attachment=True,
        download_name=filename,
        mimetype="text/plain"
    )

@app.route("/plans")
def list_plans():
    keywords = ["migration", "plan"]

    db_session = SessionLocal()

    # Build a filter that checks if AI response contains any keyword (case-insensitive)
    filters = [ChatHistory.ai_response.ilike(f"%{kw}%") for kw in keywords]

    plans = db_session.query(ChatHistory)\
        .filter(or_(*filters))\
        .order_by(ChatHistory.timestamp.desc())\
        .all()

    db_session.close()

    return render_template("plans.html", plans=plans)

if __name__ == "__main__":
    app.run(debug=True)
