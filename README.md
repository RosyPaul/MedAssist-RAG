# 🏥 MedAssist — Medical RAG Chatbot

A conversational medical chatbot powered by Retrieval-Augmented Generation (RAG). Ask any medical question and get accurate answers from a real medical knowledge base.

👉 **[Live Demo](https://rosypaul-med-bot.hf.space)**

---

## 🧠 How It Works

1. A medical textbook is chunked and embedded using HuggingFace embeddings
2. Embeddings are stored in Pinecone vector database
3. On each query, relevant chunks are retrieved from Pinecone
4. Groq (LLaMA 3.1) generates an answer based on the retrieved context

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Flask |
| LLM | Groq (LLaMA 3.1 8B) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | Pinecone |
| Orchestration | LangChain |
| Deployment | Hugging Face Spaces (Docker) 


## ☁️ AWS Deployment (CI/CD with GitHub Actions)

The app is also deployable to AWS EC2 via Docker and ECR using GitHub Actions.

**Services used:**
- **ECR** — stores the Docker image
- **EC2** — runs the container
- **GitHub Actions** — automates build and deploy on every push to `main`

**GitHub Secrets required:**
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
PINECONE_API_KEY
GROQ_API_KEY

## 📁 Project Structure
MedAssist-RAG/
├── app.py               # Flask app
├── store_index.py       # Embed and upload to Pinecone
├── Dockerfile
├── requirements.txt
├── src/
│   ├── helpers.py       # Embedding and PDF loading
│   └── prompt.py        # System prompt
└── templates/
└── chat.html        # Chat UI


