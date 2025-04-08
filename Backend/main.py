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

#-----------------------------------------------------deleted:3----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


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



#--------------------------------deleted:2--------------------------------

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


#---------------------------------------deleted:1----------------------------------











