# 🧠 RAG POC with Mistral, FAISS & FastAPI

This project is a simple Retrieval-Augmented Generation (RAG) application using:

- **Mistral 7B Instruct** (via LlamaCPP) for local LLM inference  
- **FAISS** for in-memory vector search  
- **LlamaIndex** for orchestration  
- **FastAPI** to expose the API  
- **BeautifulSoup** & `requests` for website crawling  

---

## 📁 Project Structure

```
rag_poc/
├── crawl_website.py        # Crawler script to generate knowledge base
├── docs/                   # Stores .txt files from crawled content
├── main.py                 # FastAPI app with RAG logic
├── models/                 # Directory for the mistral GGUF model
├── Dockerfile              # Container build instructions
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Prerequisites

- Docker & Docker Compose installed  
- Mistral 7B Instruct (quantized `.gguf`) downloaded into `models/`  
- Internet access to crawl the target site  

---

## 🚀 Usage

### 1. Build the Docker image

```bash
docker-compose build
```

### 2. Run the crawler (to populate `docs/`)

```bash
docker-compose run --rm rag-app python crawl_website.py https://your-site.com
```

This generates `.txt` files in the `docs/` folder for vector indexing.

### 3. Start the FastAPI server

```bash
docker-compose up
```

Once running, access the API at:

```
http://localhost:8000
```

---

## 📬 Example Query

Send a `POST` request to `/query`:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What features are available in Website app?"}'
```

---

## ✅ Notes

- You may want to commit `.gitignore` and `.dockerignore` for clean version control.  
- Add `__pycache__/`, `.DS_Store`, and `*.gguf` (if needed) to `.gitignore`.  

---

## 📄 License

MIT
