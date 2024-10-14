from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama 
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.documents import Document
import os
import numpy as np
from rake_nltk import Rake
import nltk
import pandas as pd


def store_to_df(db):
    doc_dict = db.docstore._dict
    data_rows = []
    for k in doc_dict.keys():
        content = doc_dict[k].page_content
        data_rows.append({"content": content})
    dataframe = pd.DataFrame(data_rows)
    return dataframe

def show_vstore(db):
    dataframe = store_to_df(db)
    print(dataframe)  # Use print() instead of display()

nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_dir)  # Change directory for finding nltk data
rake = Rake()

def extract_keywords(query, rake=rake):
    query = query.replace("(", "").replace(")", "")
    rake.extract_keywords_from_text(query)
    keywords = list(set(rake.get_ranked_phrases())) 
    
    return keywords

def search_with_keywords(query, database):
    keywords = extract_keywords(query)
    results = []

    for keyword in keywords:
        # Perform a similarity search for each keyword
        search_results = database.similarity_search(keyword, k=2)  # You can adjust the value of k
        results.extend(search_results)

    # Remove duplicates while preserving order
    unique_results = []
    seen = set()

    for doc in results:
        doc_id = doc.metadata.get("id", doc.page_content)  # Use a unique attribute, like ID or content
        if doc_id not in seen:
            seen.add(doc_id)
            unique_results.append(doc)

    return unique_results

# Initialize embeddings and database
embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-large-en-v1.5",
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)


db_path = os.path.join(os.path.dirname(__file__), "vector_database")
def load(db_path=db_path):
    vector_data_base = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True, normalize_L2=True)
    return vector_data_base

vector_data_base = load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

template = """Answer the users stock related questions using the real time/Yahoo Finance data provided. Take into consideration the user cannot see the context you are provided and 
    when quoting context cite what the data is instead of saying "in the data provided" etc. Give a detailed answer, up to 5 sentences.
    If you don't know the answer, say you don't know. 
    Context: {context}
    Questions: {question}
    Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

llm = Ollama(model="llama3.2:1b", base_url="https://rag-api-cpg2hhagatdabraj.eastus-01.azurewebsites.net")
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_data_base.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

async def use_RAG_pipeline(query):
    global vector_data_base 
    
    
    # Perform enhanced context retrieval
    enhanced_context = search_with_keywords(query, vector_data_base)

    if not enhanced_context:
        yield {"event": "error", "message": "No relevant information found"}
        return  # Exit the generator if no context is found

    # Combine the retrieved documents' content to form a context string
    context = "\n".join([doc.page_content for doc in enhanced_context])
    

    # Prepare the input for the LLM
  
    async for render in qa_chain.astream_events({"context": context, "query": query}, version="v2"):
        event_type = render['event']
        

        #if event_type == "on_chain_end":
        #    print(render)
        #    yield {
        #        "event_type": event_type, 
        #        "content": [doc.dict() for doc in render["data"]["output"]]
        #    }
        if event_type == "on_llm_stream":
            
            yield {
                "event_type": event_type,
                "content": render["data"]["chunk"].text
            }
    yield {
        "event_type": "done"
    }


   
    
def refresh_retriever(database:FAISS):
    retriever = database.as_retriever()
    global llm
    global qa_chain
   
    qa_chain=RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT},
    )
    
    


async def add_articles(ticker, data, data_base):
    documents=[]
   
    for title in data.keys():
        
        document=Document(
        page_content=f"Title:{title} Article: {data[title]}",
        metadata={"type": "article",
            "title": title ,
            "ticker": ticker},
        )
        documents.append(document)
    documents=text_splitter.split_documents(documents)
   
    temporary=FAISS.from_documents(documents,embeddings)
    
    data_base.merge_from(temporary)

    

    return data_base

async def add_data(ticker, data, data_base):
    documents=[]
    for variable, value in data.items():
        
        document=Document(
            page_content=f"{variable} of {ticker} is: {value}",
            metadata={"type": "data",
                        "title": variable, 
                        "ticker": ticker},
        )
        documents.append(document)

    
    temporary=FAISS.from_documents(documents,embeddings)
   
    data_base.merge_from(temporary)
    
  

    return data_base
    

async def add_tables(ticker, title, data, data_base):
    documents=[]
    
   
    document=Document(
        page_content=f"{title} of {ticker} is this table: {data}",
        metadata={"type": "table",
                    "title": title ,
                    "ticker": ticker},
    )
    documents.append(document)
    temporary=FAISS.from_documents(documents,embeddings)
    data_base.merge_from(temporary)

    
    return data_base


async def delete_data(stock, data_base):
   
  
    
    db_dict=data_base.docstore._dict
    delete=set()
    for key in db_dict.keys():
        print(db_dict[key].metadata["ticker"])
        if db_dict[key].metadata["ticker"]==stock:
            print(db_dict[key].metadata["ticker"],"matches this", stock)
            delete.add(key)
    data_base.delete(list(delete))
    refresh_retriever(data_base)
    
    return data_base



def is_rag_usefull(query, data_base):
    results = data_base.similarity_search_with_relevance_scores(
        f"{query}",
        k=1,
    )
    for res, score in results:
        
        if score < 0.5:
            return False
        return True
##Return type of similarity_search_with_score():
##List[Tuple[Document, float]]
##float>=0 and <=1, the closer to 0 the less similar it is



