from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama 
from langchain.chains.retrieval_qa.base import RetrievalQA
#from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.documents import Document
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv() # Load environment variables from.env file
'''
def download_all_nltk_data():
    # Define the directory where nltk_data should be located
    nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")

    # Check if nltk_data directory exists, if not, create it
    if not os.path.exists(nltk_data_dir):
        print(f"Creating nltk_data directory at {nltk_data_dir}...")
        os.makedirs(nltk_data_dir, exist_ok=True)

    # Add the nltk_data directory to nltk's data path
    nltk.data.path.append(nltk_data_dir)

    # Download all NLTK data
    print(f"Downloading all NLTK data to {nltk_data_dir}...")
    nltk.download('all', download_dir=nltk_data_dir)
    print("All NLTK data downloaded.")

# Call the function to ensure all nltk_data is downloaded
download_all_nltk_data()

# Initialize Rake after ensuring resources are available'''

'''
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
'''

# Initialize embeddings and database
embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-large-en-v1.5",
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)
import os

client = QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_SECRET_KEY")
)

#db_path = os.path.join(os.path.dirname(__file__), "vector_database")

def load():

    vector_data_base = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding=embeddings,
    )
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

llm = Ollama(model="llama3.2:1b", base_url=os.getenv("LLAMA_URL"))
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_data_base.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

async def use_RAG_pipeline(query, context):
   
    
    
    # Perform enhanced context retrieval
    if not context:
        yield {"event_type": "error", "message": "No relevant information found"}
        return  # Exit the generator if no context is found

    # Combine the retrieved documents' content to form a context string
    context = "\n".join([doc.page_content for doc in context])
    

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


   
    
def refresh_retriever():
    global vector_data_base
    retriever = vector_data_base.as_retriever()
    global llm
    global qa_chain
   
    qa_chain=RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT},
    )
    
    


async def add_articles(ticker, data):

    global vector_data_base
    
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
    #ids = [str(uuid4()) for _ in range(len(documents))]
    #temporary=FAISS.from_documents(documents,embeddings)
    
    #data_base.merge_from(temporary)
    ids=vector_data_base.add_documents(documents=documents)
    print("added articles")

    return ids

    

async def add_data(ticker, data):
    global vector_data_base

    documents=[]
    for variable, value in data.items():
        
        document=Document(
            page_content=f"{variable} of {ticker} is: {value}",
            metadata={"type": "data",
                        "title": variable, 
                        "ticker": ticker},
        )
        documents.append(document)

    
    #temporary=FAISS.from_documents(documents,embeddings)
   
    #data_base.merge_from(temporary)
    #documents=text_splitter.split_documents(documents)
    ids=vector_data_base.add_documents(documents)
    print("added data")

    
    return ids
    

async def add_tables(ticker, title, data):
    documents=[]
    global vector_data_base
   
    document=Document(
        page_content=f"{title} of {ticker} is this table: {data}",
        metadata={"type": "table",
                    "title": title ,
                    "ticker": ticker},
    )
    documents.append(document)
    #documents=text_splitter.split_documents(documents)
    #temporary=FAISS.from_documents(documents,embeddings)
    #data_base.merge_from(temporary)
    ids=vector_data_base.add_documents(documents)
    print("added tables")

    return ids


async def delete_data(ids):
    
    global vector_data_base
    '''db_dict=vector_data_base.docstore._dict
    delete=set()
    for key in db_dict.keys():
        print(db_dict[key].metadata["ticker"])
        if db_dict[key].metadata["ticker"]==stock:
            print(db_dict[key].metadata["ticker"],"matches this", stock)
            delete.add(key)
    data_base.delete(list(delete))
    refresh_retriever(data_base)'''

    vector_data_base.delete(ids)
    
    return 



def is_rag_usefull(query):

    global vector_data_base

    results = vector_data_base.similarity_search_with_relevance_scores(
        query,
        k=4                                                                 
    )
    context=[]

    for res, score in results:
        if score > 0.79:
            context.append(res)


    if len(context)<2: #if more than half of the results have low relevance to the prompt, then we don't have much data for the
        return False, context
    return True, context
##Return type of similarity_search_with_score():
##List[Tuple[Document, float]]
##float>=0 and <=1, the closer to 0 the less similar it is

