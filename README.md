

# **Job Recommendation System**  
**Scraping SHL Data → Embeddings → ChromaDB → Mistral API → Cosine Similarity → Frontend UI**  

This project scrapes job-related data from SHL, processes it into embeddings, stores them in ChromaDB, and uses Mistral API to recommend the top 5 job profiles based on a user-provided job description.  

## **🚀 Features**  
- **Web Scraping**: Extracts job data from SHL (or provided dataset).  
- **Embeddings**: Converts job descriptions into vector embeddings.  
- **Vector Database**: Stores embeddings in ChromaDB for fast retrieval.  
- **Mistral API**: Computes cosine similarity for recommendations.  
- **Frontend UI**: Simple interface to input a job description and get matches.  

---

## **🛠️ How It Works**  

### **1. Data Collection (Scraping)**
- Used Python (`requests`, `BeautifulSoup`, or `Scrapy`) to scrape job profiles from SHL.  
- Extracted: **Job Title, Description, Skills, Experience Level**.  
- Stored in `JSON/CSV` for further processing.  

### **2. Embedding Generation**
- Used **Sentence Transformers** (`all-MiniLM-L6-v2`) or OpenAI embeddings to convert text into vectors.  
- Example code:  
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model.encode(job_descriptions)
  ```

### **3. Storing in ChromaDB**
- Created a ChromaDB collection to store embeddings with metadata (job title, URL, etc.).  
  ```python
  import chromadb
  client = chromadb.PersistentClient(path="job_embeddings_db")
  collection = client.create_collection(name="job_profiles")
  collection.add(embeddings=embeddings, documents=job_descriptions, metadatas=metadata)
  ```

### **4. Mistral API for Recommendations**
- When a user submits a job description:  
  1. Convert input into an embedding.  
  2. Query ChromaDB for nearest matches (cosine similarity).  
  3. Use Mistral API to refine & rank top 5 jobs.  
  ```python
  def get_recommendations(query_embedding):
      results = collection.query(query_embeddings=[query_embedding], n_results=5)
      return results["documents"]
  ```

### **5. Frontend (Streamlit/Gradio/Flask)**
- Simple UI where users input a job description.  
- Backend processes the query → returns top 5 matches.  
- Example (Streamlit):  
  ```python
  import streamlit as st
  st.text_input("Enter Job Description")
  if st.button("Recommend Jobs"):
      recommendations = get_recommendations(user_input)
      st.write(recommendations)
  ```

---

## **⚙️ Setup & Run**  

### **Prerequisites**  
- Python 3.10+  
- Libraries: `chromadb`, `sentence-transformers`, `mistralai`, `requests`, `BeautifulSoup`  

### **Installation**  
```bash
git clone https://github.com/yourusername/job-recommender.git
cd job-recommender
pip install -r requirements.txt
```

### **Run the Project**  
1. **Scrape Data** (if needed):  
   ```bash
   python scraper.py
   ```
2. **Generate & Store Embeddings**:  
   ```bash
   python embeddings.py
   ```
3. **Launch Frontend**:  
   ```bash
   streamlit run app.py
   ```

---
