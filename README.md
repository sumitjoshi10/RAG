# RAG

This project demonstrates how to perform **semantic search with similarity scores** using **LangChain embeddings** and **scikit-learn cosine similarity**, **without using any vector database** such as FAISS, Chroma, or Pinecone.

This approach is ideal for:
- Small to medium-sized documents
- Debugging and learning embeddings
- Lightweight or custom pipelines
- Avoiding external infrastructure

---

## 🚀 Features

- Load `.txt` documents
- Split text into semantic chunks
- Generate embeddings using LangChain
- Perform semantic search using cosine similarity
- Rank results with similarity scores

---

## 🧱 Tech Stack

- **Python**
- **LangChain** – document loading, chunking, embeddings
- **Sentence Embedding**
- **scikit-learn** – cosine similarity
- **NumPy** – vector operations

---

## 📁 Project Structure
```text
RAG/
│
├── api/ # API or inference scripts
├── artifacts/ # Trained model weights
├── config/ # Configuration files
├── data/ # Dataset and annotations
├── experiment/ # Training experiments and logs
├── src/
│ └── rag/
│     └── Components
|     └── Configuration
|     └── Constants
|     └── Pipeline
|     └── Utils
├── requirements.txt
├── setup.py
├── test.py
├── README.md # Project documentation
└── LICENSE # Apache License 2.0
```

## ⚙️ Installation & Setup

1️⃣ Clone the repository

``` bash
git clone https://github.com/sumitjoshi10/RAG.git
cd RAG
```
2️⃣ Create a virtual environment (optional but recommended)
``` bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

4️⃣ Run the application

``` bash
python test.py
```

------------------------------------------------------------------------