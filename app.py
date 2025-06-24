import os
import boto3
import requests

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
from werkzeug.utils import secure_filename
from botocore.session import Session

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


import requests

def get_azure_vm_price(vm_size="Standard_D2s_v3", region="eastus"):
    url = "https://prices.azure.com/api/retail/prices"
    base_filter = (
        f"serviceName eq 'Virtual Machines' and "
        f"armRegionName eq '{region}' and "
        f"priceType eq 'Consumption'"
    )

    # Step 1: Try exact match
    exact_params = {
        "$filter": base_filter + f" and skuName eq '{vm_size}'"
    }

    try:
        res = requests.get(url, params=exact_params)
        data = res.json()
        items = data.get("Items", [])
        print("Request URL:", res.url)
        print("Items found:", len(items))

        if items:
            sorted_items = sorted(items, key=lambda x: x.get("retailPrice", float('inf')))
            best_price = sorted_items[0]
            return f"${best_price['retailPrice']} {best_price['currencyCode']}/hour for {vm_size} in {region}"

        # Step 2: If not found, fallback to related SKUs (e.g., Standard_D2s*)
        fallback_params = {
            "$filter": base_filter  # broader search
        }

        fallback_res = requests.get(url, params=fallback_params)
        fallback_data = fallback_res.json()
        fallback_items = fallback_data.get("Items", [])
        related = [item for item in fallback_items if vm_size.split('_')[0] in item.get("skuName", "")]

        print(f"Fallback matched {len(related)} related SKUs")

        if related:
            sorted_related = sorted(related, key=lambda x: x.get("retailPrice", float('inf')))
            alt = sorted_related[0]
            return f"ℹ️ Fallback price: ${alt['retailPrice']} {alt['currencyCode']}/hr for {alt['skuName']} in {region}"

        return f"Azure: Price not found for {vm_size} in {region}, no fallback available"

    except Exception as e:
        print(f"Error fetching Azure price: {e}")
        return "Azure: Error retrieving price"


def get_aws_ec2_price(instance_type="t3.micro", region="US East (N. Virginia)"):
    try:
        # Optional: Print to confirm it's loaded (remove in prod)
        print("Access Key:", os.getenv("AWS_ACCESS_KEY_ID"))

# Initialize boto3 client
        #pricing_client = boto3.client("pricing")
        client = boto3.client("pricing")

        response = client.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "location", "Value": region},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"}
            ],
            MaxResults=1
        )

        price_item = response['PriceList'][0]
        import json
        price_data = json.loads(price_item)

        price_dimensions = list(price_data['terms']['OnDemand'].values())[0]['priceDimensions']
        price = list(price_dimensions.values())[0]['pricePerUnit']['USD']
        return f"${price}/hour for {instance_type} in {region}"
    except Exception as e:
        print(f"Error fetching AWS price: {e}")
        return "Error retrieving price"


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agent", methods=["POST"])
def agent_chat():
    user_input = request.json.get("message", "")
    docs = db.similarity_search(user_input, k=10)
    context = "\n\n".join(doc.page_content for doc in docs)
    azure_price = get_azure_vm_price()
    aws_price = get_aws_ec2_price()

    keywords = ["migration", "cloud", "aws", "azure", "infrastructure", "plan", "roadmap", "rehost", "replatform"]
    is_migration_request = any(word in user_input.lower() for word in keywords)

    if is_migration_request:
        system_prompt = f"""
You are a senior cloud architect tasked with creating a structured, realistic 3-year migration plan to AWS and Azure.

Use only the data below as your source of truth:
\n\n{context}\n\n

The uploaded documents include:
- `application_dependencies.csv`: Lists app-to-app and app-to-db relationships.
- `applications.csv`: Metadata about deployed apps (name, function, owner, etc).
- `virtual_machines.csv`: Inventory of VMs (CPU, RAM, OS, workload).
- `network_links.csv`: Existing LAN/WAN connections and latency constraints.
- `physical_servers.csv`: Legacy servers that may need rehosting or refactoring.
- `storage_volumes.csv`: Attached storage volumes and sizes.
- `.pdf` files: Contain lease expiration details that can guide facility transitions.

🧠 Based on this, your plan should:
1. Group applications by dependency (from `application_dependencies.csv`) and prioritize tightly coupled services together.
2. Recommend Azure or AWS instance types that match existing VMs.
3. Plan for storage and network transitions using attached volume and link data.
4. Take into account lease expiration timelines from PDFs to inform migration waves.
5. Include fallback strategies for apps on legacy hardware that may not rehost well.

📅 Structure the migration plan in 3 phases:
- **Year 1:** Discovery, Assessment, Proof of Concept (POC)
- **Year 2:** Lift-and-Shift and Hybrid Operation
- **Year 3:** Optimization and Cloud-Native Refactoring

Include estimated cloud costs when possible using the following:
- Azure VM (Standard_B1s in eastus): **{azure_price}**
- AWS EC2 (t3.micro in US East): **{aws_price}**

When recommending cloud services, include **Markdown links** to:
- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [AWS Pricing Calculator](https://calculator.aws.amazon.com/)

Your response must be professional, actionable, and clearly formatted using Markdown (headings, tables, bullet points, etc.).
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
        temperature=0.5,
        max_tokens=1000
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
    "plan_id": plan_id if is_migration_request else None,
    "aws_price": aws_price,
    "azure_price": azure_price
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
        temperature=0.5,
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
def upload_files():
    files = request.files.getlist("file")
    uploaded_files = []
    saved_paths = []

    if not files:
        return jsonify({
            "status": "error",
            "message": "❌ No files uploaded"
        }), 400

    for file in files:
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            full_path = os.path.normpath(file.filename)
            secure_path = secure_filename(full_path)
            destination_path = os.path.join(UPLOAD_FOLDER, secure_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            if os.path.exists(destination_path):
                print(f"⚠️ Skipping duplicate file: {secure_path}")
                continue

            file.save(destination_path)
            uploaded_files.append(file.filename)
            saved_paths.append(destination_path)
        else:
            return jsonify({
                "status": "error",
                "message": f"❌ Unsupported file type: {file.filename}"
            }), 400

    try:
        # ✅ Index each file one at a time using incremental FAISS updates
        global db
        for path in saved_paths:
            db = create_faiss_index(incremental=True, new_file_path=path)

        # ✅ Update document count
        db_session = SessionLocal()
        stats = db_session.query(MigrationStats).first()
        if not stats:
            stats = MigrationStats()
            db_session.add(stats)
            stats.documents_uploaded = len(uploaded_files)
        else:
            stats.documents_uploaded = (stats.documents_uploaded or 0) + len(uploaded_files)
        db_session.commit()
        db_session.close()

        return jsonify({
            "status": "success",
            "message": f"✅ Uploaded and indexed: {', '.join(uploaded_files)}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"❌ Error indexing documents: {str(e)}"
        }), 500

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
    filters = [ChatHistory.user_input.ilike(f"%{kw}%") for kw in keywords]

    plans = db_session.query(ChatHistory)\
        .filter(or_(*filters))\
        .order_by(ChatHistory.timestamp.desc())\
        .all()

    db_session.close()

    return render_template("plans.html", plans=plans)




if __name__ == "__main__":
    app.run(debug=True)
