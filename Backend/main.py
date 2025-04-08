# main.py
import json
import os

import chromadb
from mistralai import Mistral
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI
from pymongo.errors import PyMongoError
from fastapi.middleware.cors import CORSMiddleware





app = FastAPI()
#------------------------------------ change :1 
# CORS origins
CORS_ORIGINS = [
    "http://localhost:3000",
    "https://final-front-qoan.onrender.com",
]
#--------------------------------------------Altered code: chage :1

# CORS_ORIGINS = [
#     "https://shl-frontend.onrender.com",  # Add your frontend URL
#     "http://localhost:3000",              # For local dev
# ]
#------------------------------------------------Change : 2 Added folowing code which was not in original code.






if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

#----------------------------------------------------------



# CORS middleware setup function
def configure_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
# Configure middlewares
configure_cors(app)

# Mistral setup
api_key = "PbY0APL9sTS3FkCN1f83VasfUKrjFjAk"
model_emb = "mistral-embed"
client_emb = Mistral(api_key=api_key)

# ChromaDB setup
persist_directory = "./chroma"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection(name="sde_questions")

# Embedding helper for batch (used when needed)
def get_embeddings_by_chunks_from_df(df, chunk_size, column_name):
    data = df[column_name].dropna().astype(str).tolist()
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    embeddings_response = [
        client_emb.embeddings.create(model=model_emb, inputs=chunk)
        for chunk in chunks
    ]

    embeddings = [d.embedding for response in embeddings_response for d in response.data]
    return embeddings

# Main query function
def retrieve_assessments(job_description=None, top_n=5):
    if job_description:
        # Direct embedding for single input to avoid overhead
        embedding_response = client_emb.embeddings.create(
            model=model_emb,
            inputs=[job_description]
        )
        query_embedding = embedding_response.data[0].embedding
    else:
        query_embedding = None

    # Query vector store
    results = collection.query(query_embeddings=[query_embedding] if query_embedding else None, n_results=top_n)
    print(f"Query Results: {results}")

    # Extract metadata
    metadata_results = results.get("metadatas", [[]])[0]
    formatted_results = [
        {
            "description": meta.get("description", ""),
            "job_levels": meta.get("job_levels", []),
            "languages": meta.get("languages", [])
        }
        for meta in metadata_results
    ]

    return formatted_results

query = "give 2 job description"
print(retrieve_assessments(query))



# # Path to the JSON file
# json_path = os.path.join(os.path.dirname(__file__), 'extracted_data.json')

# # Load JSON content
# with open(json_path, 'r', encoding='utf-8') as f:
#     assessment_data = json.load(f)

# # Optional: print first entry to confirm
# print(assessment_data[0])

class JobDescriptionRequest(BaseModel):
    job_description: str
    top_n: int = 5

@app.get("/")
def root():
    return {"message": "🚀 MongoDB FastAPI is live!"}

@app.post("/get-assessments")
def get_assessments(request: JobDescriptionRequest):
    try:
        results = retrieve_assessments(job_description=request.job_description, top_n=request.top_n)
        return {"assessments": results}
    except Exception as e:
        return {"error": str(e)}



# @app.get("/insert-dummy")
# def insert_dummy():
#     if db is None:
#         return {"error": "Database connection failed"}

#     collection_name = "test_details"

#     # Create collection if not exists
#     if collection_name not in db.list_collection_names():
#         db.create_collection(collection_name)

#     collection = db[collection_name]


#     try:
#         result = collection.insert_many(assessment_data)
#         return {"inserted_ids": [str(_id) for _id in result.inserted_ids]}
#     except PyMongoError as e:
#         return {"error": str(e)}

#-------------------------------------------------------------------


# # main.py
# import os
# import json
# import pandas as pd
# import chromadb
# from mistralai import Mistral
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # Initialize FastAPI app
# app = FastAPI()

# # CORS setup to allow requests from React frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with ["http://localhost:3000"] to restrict
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mistral setup
# api_key = "PbY0APL9sTS3FkCN1f83VasfUKrjFjAk"
# model_emb = "mistral-embed"
# client_emb = Mistral(api_key=api_key)

# # ChromaDB setup
# persist_directory = "./chroma"
# client = chromadb.PersistentClient(path=persist_directory)
# collection = client.get_collection(name="sde_questions")

# # Request model
# class JobDescriptionRequest(BaseModel):
#     job_description: str
#     top_n: int = 5

# # Main retrieval logic
# def retrieve_assessments(job_description=None, top_n=5):
#     if job_description:
#         embedding_response = client_emb.embeddings.create(
#             model=model_emb,
#             inputs=[job_description]
#         )
#         query_embedding = embedding_response.data[0].embedding
#     else:
#         query_embedding = None

#     results = collection.query(query_embeddings=[query_embedding] if query_embedding else None, n_results=top_n)

#     metadata_results = results.get("metadatas", [[]])[0]
#     formatted_results = [
#         {
#             "name": meta.get("name", "Untitled Assessment"),
#             "url": meta.get("url", "#"),
#             "description": meta.get("description", ""),
#             "duration": meta.get("duration", "Unknown"),
#             "remote": meta.get("remote", "No"),
#             "adaptive": meta.get("adaptive", "No"),
#             "type": meta.get("type", "General")
#         }
#         for meta in metadata_results
#     ]

#     return formatted_results

# # Health check
# @app.get("/")
# def root():
#     return {"message": "🚀 SHL API is live!"}

# # Main endpoint for frontend to call
# @app.post("/get-assessments")
# def get_assessments(request: JobDescriptionRequest):
#     try:
#         results = retrieve_assessments(job_description=request.job_description, top_n=request.top_n)
#         return {"assessments": results}
#     except Exception as e:
#         return {"error": str(e)}










