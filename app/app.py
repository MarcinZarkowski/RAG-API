from fastapi import FastAPI, Depends, Body, HTTPException, Request, WebSocket
import json
from sqlalchemy.orm import Session
from .models import SessionLocal, Chat
from datetime import datetime, timedelta, timezone
import requests
from .get_stock_data import *
from .Rag_handler import add_articles, add_tables, add_data, use_RAG_pipeline, delete_data, load, refresh_retriever, show_vstore, is_rag_usefull, extract_keywords
import pandas as pd
import nltk
import os

from dotenv import load_dotenv



load_dotenv()
app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 4000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)

    
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

    

@app.middleware("http")
async def log_request(request: Request, call_next):
    try:
        body = await request.body()
        # You can log or inspect the request body here if needed
    except Exception as e:
        print(f"Error reading body: {e}")
    
    response = await call_next(request)
    return response



@app.post("/start-chat")
async def start_conversation(body: dict = Body(...), db: Session = Depends(get_db)):  
    global vector_data_base
    key=body.get("key")
    if key!= os.getenv("KEY"):
        return {"message": "Invalid key"}
    
    show_vstore(vector_data_base)
    stock = body.get("stock")
    db_chat = db.query(Chat).filter(Chat.title == stock).first()

    if db_chat:
        # Ensure created_at is timezone-aware
        created_at = db_chat.created_at
        if created_at.tzinfo is None: 
            created_at = created_at.replace(tzinfo=timezone.utc) 

        # Check if the chat is older than 12 hours
        if datetime.now(timezone.utc) - created_at > timedelta(hours=12): 
            db.delete(db_chat)
            db.commit()
            print("TOO OLD, Making new")
            vector_data_base=await delete_data(stock, vector_data_base)
            

            #get new data
            context = await get_stock_context(stock)
            
            vector_data_base= await add_stock_context(stock, context, db) 
            refresh_retriever(vector_data_base)
            show_vstore(vector_data_base)
            
    else:
        context = await get_stock_context(stock)
        vector_data_base= await add_stock_context(stock, context, db) 
        refresh_retriever(vector_data_base)
        show_vstore(vector_data_base)
       
    
        
    
     

async def add_stock_context(stock, context, db: Session):
    global vector_data_base
    try:
        db_chat = db.query(Chat).filter(Chat.title == stock).first()
        if not db_chat:
            
         
            for key, value in context.items():
                if key == "news":
                    vector_data_base = await add_articles(stock, value, vector_data_base)
                elif key == "calendar" or key == "analyst_price_targets":
                    vector_data_base = await add_data(stock, value, vector_data_base)
                else:
                    vector_data_base = await add_tables(stock, key, value, vector_data_base)

            save_directory="app/vector_database"
            vector_data_base.save_local(save_directory)

            #only save to sqlite once vectordb processes completed
            context_json = json.dumps(context, default=str)  
            db_chat = Chat(title=stock, context=context_json)
            db.add(db_chat)
            db.commit()
            db.refresh(db_chat)
    
        
        return vector_data_base

    except Exception as e:
        print(f"Error in add_stock_context: {e}")
        return vector_data_base
'''      
#@app.post("/ask")
async def ask(body: dict=Body(...), db: Session = Depends(get_db)):
    key=body.get("key")
    # Initialize environment variables
    
    if key!= os.getenv("KEY"):
        
        return {"message": "Invalid key"}
    ticker=body.get("ticker")
    
    db_chat = db.query(Chat).filter(Chat.title == ticker).first()
    if not db_chat:
        return {"message": "No chat found"}
    prompt=body.get("prompt") + f"(This prompt is about {ticker})"
    
    #if is_rag_usefull(prompt, vector_data_base):
    response=use_RAG_pipeline(prompt)
    #else:
       # response="Sorry, either we don't have enough information on that, or the prompt is not relevant."

    print(response)
    
    return {"message": response}'''

@app.websocket('/ask')
async def send_chat(websocket: WebSocket):
    global vector_data_base
    show_vstore(vector_data_base)
    await websocket.accept()
   
    while True:
        data = await websocket.receive_text()
        parsed_data = json.loads(data)  # Parse the incoming JSON
        key = parsed_data.get("key")
        if key!= os.getenv("KEY"):
            return {"message": "Invalid key"}
        question = parsed_data.get("query")
        ticker = parsed_data.get("ticker")
        
        should_we_ask=False
        for word in extract_keywords(question):
            if is_rag_usefull(word, vector_data_base):
                should_we_ask=True
                break

        if not should_we_ask:
            await websocket.send_text(json.dumps({"event_type": "bad_request"}))
            await websocket.close()
            return

        should_we_ask=is_rag_usefull(question, vector_data_base)
        
        
        question= question + f"(This prompt is about {ticker})"
       

        async for event in use_RAG_pipeline(question):
           
            if event["event_type"] == "done":
                await websocket.close()
                return 
            else:
                await websocket.send_text(json.dumps(event))
