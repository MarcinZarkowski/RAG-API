from sqlalchemy import Column, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import ARRAY
import os
from dotenv import load_dotenv
 
load_dotenv()

Base = declarative_base()

class Chat(Base):
    __tablename__ = 'chat'

    title = Column(String, primary_key=True)
    #context = Column(String, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ids = Column(String)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")#"sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)

