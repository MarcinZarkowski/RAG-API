#RAG-API

-This API creates a rag pipline using the huggingface module in Python fastAPI.

-Two databases are used, an sqlite database keeps track of data collected for each stock requested with a time stamp. If a user requests a stock and it doesn't exist or it is older than 12 hours, the api collects updated stock data, deletes old stock information in the vector database, and adds the new information.

-NLTK module is used to extract keywords from a user query to make sure query is relevant or can be answered by data in the vector database. If not, it returns a default message saving resources and time. 

-Docker was used to deploy on Azure, NLTK was downloaded during build process to cut down startup times. 
