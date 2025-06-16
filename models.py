from sqlalchemy import Column, Integer, String, Text, DateTime, func, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# ✅ Use a single Base definition
Base = declarative_base()

# 🧠 Tracks migration metrics
class MigrationStats(Base):
    __tablename__ = "migration_stats"

    id = Column(Integer, primary_key=True, index=True)
    plans_generated = Column(Integer, default=0)
    documents_uploaded = Column(Integer, default=0)
    user_messages = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# 💬 Stores chat history
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64))  # Unique identifier per chat session
    user_input = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# 🗃️ SQLite setup
engine = create_engine("sqlite:///chat_history.db")
Base.metadata.create_all(engine)

# 🔁 Session factory
SessionLocal = sessionmaker(bind=engine)
