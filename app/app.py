from fastapi import FastAPI, Depends, Body, HTTPException, Request, WebSocket
import json
from sqlalchemy.orm import Session
from .models import SessionLocal, Chat
from datetime import datetime, timedelta, timezone
import requests
from .get_stock_data import *
from .Rag_handler import add_articles, add_tables, add_data, use_RAG_pipeline, delete_data, refresh_retriever, is_rag_usefull
import pandas as pd

import os

from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
'''
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 4000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)'''

    
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
    #global vector_data_base
    key=body.get("key")

    if key!= os.getenv("KEY"):
        return {"message": "Invalid key"}
    
    #show_vstore(vector_data_base)
    stock = body.get("stock")
    db_chat = db.query(Chat).filter(Chat.title == stock).first()

    if db_chat:
        # Ensure created_at is timezone-aware
        created_at = db_chat.created_at
        if created_at.tzinfo is None: 
            created_at = created_at.replace(tzinfo=timezone.utc) 

        # Check if the chat is older than 12 hours
        if datetime.now(timezone.utc) - created_at > timedelta(hours=12): 
            ids=json.loads(db_chat.ids)
            print("TOO OLD, Making new")
            await delete_data(ids) #wait until data is deleted

            db.delete(db_chat) #delete chat from db
            db.commit()
            print("deleted")

            #get new data
            context = await get_stock_context(stock)
            
            await add_stock_context(stock, context, db) #wait to refresh retreiver until data is added
            refresh_retriever()
            #show_vstore(vector_data_base)
        else:
            print("NOT TOO OLD")
    
    else:
        context = await get_stock_context(stock)
        await add_stock_context(stock, context, db) 
        refresh_retriever()
        #show_vstore(vector_data_base)
       
    
        
    
     
#creates SQlite instance of a conversation about a stock and stores its context, Document ids to then easily delete from vector databse and timestamp 
#to keep track of how old the context is
async def add_stock_context(stock, context, db: Session):
    #global vector_data_base
    
    try:
        db_chat = db.query(Chat).filter(Chat.title == stock).first()
        if not db_chat:#check one more time if chat exists
            ids=[]
         
            for key, value in context.items():
                if key == "news":
                    ids_just_added = await add_articles(stock, value)
                    ids.extend(ids_just_added)
                elif key == "calendar" or key == "analyst_price_targets":
                    ids_just_added= await add_data(stock, value )
                    ids.extend(ids_just_added)
                else:
                    ids_just_added = await add_tables(stock, key, value)
                    ids.extend(ids_just_added)

            '''save_directory="app/vector_database"
            vector_data_base.save_local(save_directory)'''


            #only save to sqlite once vectordb processes completed
            #context_json = json.dumps(context, default=str)  
            ids=json.dumps(ids)
            db_chat = Chat(title=stock, ids=ids)#context=context_json
            db.add(db_chat)
            db.commit()
            db.refresh(db_chat)
    
        
        return 

    except Exception as e:
        print(f"Error in add_stock_context: {e}")
        return 
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
    #global vector_data_base
    #show_vstore(vector_data_base)
    await websocket.accept()
   
    while True:
        data = await websocket.receive_text()
        parsed_data = json.loads(data)  # Parse the incoming JSON
        key = parsed_data.get("key")
        if key!= os.getenv("KEY"):
            return {"message": "Invalid key"}
        question = parsed_data.get("query")
        ticker = parsed_data.get("ticker")
        
        
        usefull, context= is_rag_usefull(question)
        if not usefull:
            await websocket.send_text(json.dumps({"event_type": "bad_request"}))
            await websocket.close()
            return
             
        print("got after check")
        
        question= question + f"(This prompt is about {ticker})"

        async for event in use_RAG_pipeline(question, context):
            if event["event_type"]=="error":
                await websocket.send_text(json.dumps({"event_type": "bad_request"}))
                await websocket.close()
                return
            elif event["event_type"] == "done":
                await websocket.close()
                return 
            else:
                await websocket.send_text(json.dumps(event))
