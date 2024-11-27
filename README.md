#RAG-API - Demo of Complete WebApp using this API: https://www.youtube.com/watch?v=WLgpq9QTqIY

-This API creates a rag pipline using the huggingface module in Python fastAPI.

-Two databases are used, one postgress D.B. keeps track of data collected for each stock requested with a time stamp. If a user requests a stock and it doesn't exist or it is older than 12 hours, the api collects updated stock data, deletes old stock information in the vector database, and adds the new information. The Postgres database keeps a time stamp for each stock in the vector database (allowing for the 12 hour checks) and holds an array of ids to delete vectors from the vector database efficiently, when they're old. 

-Data is retrieved from vector database with a relevence score to make sure query is relevant or can be answered by data in the vector database. If not, it returns a default message saving resources and time. 

-Docker was used to deploy on Azure, NLTK was downloaded during build process to cut down startup times. 
