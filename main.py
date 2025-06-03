from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.settings import Settings
from llama_index.core.schema import Document
import faiss
import os
import sys
import torch

# -----------------------
# Initialize FastAPI
# -----------------------
app = FastAPI()

class Query(BaseModel):
    question: str

# -----------------------
# Load documents with metadata
# -----------------------
docs_dir = "docs"
file_docs = []
for filename in os.listdir(docs_dir):
    if filename.endswith(".txt"):
        path = os.path.join(docs_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        file_docs.append(Document(text=content, metadata={"filename": filename}))

# -----------------------
# Embeddings + Vector Store
# -----------------------
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", normalize=True)
faiss_index = faiss.IndexFlatIP(384)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -----------------------
# LLM: Quantized Mistral
# -----------------------
use_gpu = torch.cuda.is_available()
n_gpu_layers = 40 if use_gpu else 0

llm = LlamaCPP(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_new_tokens=512,
    context_window=2048,
    generate_kwargs={"top_p": 0.95, "frequency_penalty": 0},
    model_kwargs={"n_gpu_layers": n_gpu_layers},
    verbose=True,
)

# -----------------------
# Global Settings
# -----------------------
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# -----------------------
# Chunking with SentenceSplitter
# -----------------------
parser = SentenceSplitter(chunk_size=512)
nodes = parser.get_nodes_from_documents(file_docs)

# -----------------------
# Build the Index
# -----------------------
index = VectorStoreIndex(nodes, storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=2)

# -----------------------
# Query Endpoint
# -----------------------
@app.post("/query")
async def query_api(query: Query):
    try:
        # Perform RAG using query
        response = query_engine.query(query.question)

        if not response.source_nodes or response.source_nodes[0].score < 0.2:
            fallback_response = llm.complete(query.question)
            return {
                "response": fallback_response.text.strip(),
                "sources": []
            }

        return {
            "response": str(response),
            "sources": [
                {
                    "filename": node.node.metadata.get("filename", "unknown"),
                    "score": round(node.score, 4),
                    "snippet": node.node.text[:200] + "..."
                }
                for node in response.source_nodes
            ]
        }
    except Exception as e:
        print(f"❌ RAG failed: {e}")
        fallback_response = llm.complete(query.question)
        return {
            "response": fallback_response.text.strip(),
            "sources": []
        }
