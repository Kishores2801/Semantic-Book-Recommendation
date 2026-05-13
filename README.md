# 📚 Semantic Book Recommendation System

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI_Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge)](https://www.trychroma.com/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

> **An AI-powered semantic book recommendation engine that understands natural language queries, filters by genre and emotional tone, and returns the most contextually relevant books — built with LangChain, OpenAI Embeddings, ChromaDB vector search, and a Gradio interface.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [RAG Pipeline](#rag-pipeline)
- [Emotion-Based Tone Filtering](#emotion-based-tone-filtering)
- [Getting Started (Local)](#getting-started-local)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Example Queries](#example-queries)
- [License](#license)

---

## 🧭 Overview

This project goes **far beyond traditional keyword-based book search**. Instead of matching exact words, it uses **semantic similarity** powered by OpenAI embeddings to understand the *meaning* behind a user's query and return books that genuinely match their intent.

For example, a query like *"I want a thrilling mystery novel set in Victorian England"* returns books that semantically match the mood, theme, and setting — even if none of those exact words appear in the book description.

The system further enriches recommendations by allowing users to filter by:
- **Category** (Fiction, Non-Fiction, Mystery, Science, etc.)
- **Emotional Tone** (Happy, Sad, Angry, Suspenseful, Surprising, Neutral)

---

## 🧠 How It Works

The recommendation pipeline has two core components:

### 1. Semantic Search (Vector Similarity)
Book descriptions are embedded into high-dimensional vectors using **OpenAI Embeddings** and stored in a **ChromaDB** vector database. When a user submits a query, it is embedded with the same model and the most semantically similar books are retrieved via cosine similarity search.

### 2. Emotion-Based Re-Ranking
Each book in the dataset has been pre-labelled with emotion scores (`joy`, `sadness`, `anger`, `fear`, `surprise`, `neutral`) derived from the book descriptions. After semantic retrieval, results are re-ranked by the selected emotional tone to surface the most emotionally aligned books.

---

## ✨ Features

- 🔍 **Natural Language Search** — Describe what you want in plain English; no keywords needed
- 🧠 **OpenAI Semantic Embeddings** — Deep contextual understanding of book descriptions
- 🗄️ **ChromaDB Vector Store** — Fast similarity search across the full book catalog
- 😊 **Emotion Tone Filtering** — Sort results by emotional tone: Happy, Sad, Angry, Suspenseful, Surprising, or Neutral
- 📂 **Category Filtering** — Filter by genre/category (Fiction, Non-Fiction, etc.)
- 🖼️ **Visual Book Gallery** — Results displayed as a Gradio gallery with cover images and descriptions
- 📖 **Smart Author Formatting** — Handles single, dual, and multi-author formatting gracefully
- ⚡ **Top-K Retrieval** — Fetches top 50 semantic matches, then filters down to top 16 for quality

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| **Language** | Python | 3.x |
| **LLM Orchestration** | LangChain Community | 0.4.1 |
| **Text Splitting** | LangChain Text Splitters | 1.1.1 |
| **Embeddings** | LangChain OpenAI (OpenAI Embeddings) | 1.1.12 |
| **Vector Store** | ChromaDB + LangChain Chroma | 1.5.5 / 1.1.0 |
| **LLM Provider** | OpenAI API | 2.31.0 |
| **Tokenisation** | tiktoken | 0.12.0 |
| **Web Interface** | Gradio | 6.10.0 |
| **Data Manipulation** | Pandas | 3.0.1 |
| **Numerical Computing** | NumPy | 2.4.3 |
| **Env Management** | python-dotenv | 1.2.2 |

---

## 📁 Project Structure

```
Semantic-Book-Recommendation/
├── .github/workflows/          # GitHub Actions CI/CD workflows
├── .vscode/                    # VS Code workspace settings
├── data/
│   └── processed/
│       ├── book_categorized_with_emotions.csv  # Books with category + emotion scores
│       └── tagged_description.txt              # Book descriptions for vector indexing
├── notebooks/                  # Jupyter Notebooks for EDA, emotion labelling & preprocessing
├── src/                        # Modular Python source code
├── app.py                      # Main Gradio application (entry point)
├── requirements.txt            # Pinned Python dependencies
├── runtime.txt                 # Python runtime version specification
├── .gitignore                  # Git ignore rules
└── README.md
```

---

## 🔄 RAG Pipeline

This project implements a **RAG (Retrieval-Augmented Generation)** style architecture for book recommendations:

```
User Query (natural language)
        │
        ▼
┌──────────────────────────┐
│  OpenAI Embeddings       │  → Converts query to dense vector
│  (text-embedding model)  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  ChromaDB Vector Search  │  → Finds top-50 semantically
│  (cosine similarity)     │     similar book descriptions
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Category Filter         │  → Optional genre filter
│  (simple_categories)     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Emotion Re-Ranking      │  → Sort by joy / sadness /
│  (pre-labelled scores)   │     anger / fear / surprise
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Gradio Gallery UI       │  → Top 16 results with cover
│                          │     images + descriptions
└──────────────────────────┘
```

---

## 😊 Emotion-Based Tone Filtering

Each book has been pre-processed with **emotion scores** derived from its description. The available tones and their underlying emotion columns are:

| UI Tone | Emotion Column | Description |
|---|---|---|
| **Happy** | `joy` | Uplifting, feel-good books |
| **Sad** | `sadness` | Emotional, melancholic reads |
| **Angry** | `anger` | Intense, confrontational themes |
| **Suspenseful** | `fear` | Thriller, horror, tension-driven |
| **Surprising** | `surprise` | Unexpected twists, unconventional narratives |
| **Neutral** | `neutral` | Balanced, informational, or matter-of-fact tone |

---

## 🚀 Getting Started (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/Kishores2801/Semantic-Book-Recommendation.git
cd Semantic-Book-Recommendation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the App

```bash
python app.py
```

Open **[http://localhost:10000](http://localhost:10000)** in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key for generating embeddings |

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## ☁️ Deployment

This project is configured for deployment on **Hugging Face Spaces** using the Gradio SDK, as defined in the `README` metadata:

```yaml
title: Book_Recommendation_System
app_file: app.py
sdk: gradio
sdk_version: 6.10.0
```

To deploy on Hugging Face Spaces:
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Push this repository to the Space
4. Add `OPENAI_API_KEY` as a Secret in the Space settings

> The app can also run locally or on any server that supports Python and exposes port 10000.

---

## 💡 Example Queries

Try these in the search box:

| Query | Category | Tone |
|---|---|---|
| `"A gripping psychological thriller with an unreliable narrator"` | Fiction | Suspenseful |
| `"Inspiring stories of people overcoming adversity"` | Non-Fiction | Happy |
| `"A heartbreaking love story set during wartime"` | Fiction | Sad |
| `"Mind-bending science fiction about artificial intelligence"` | Fiction | All |
| `"A cosy mystery in a small English village"` | Fiction | Neutral |

---

## 📄 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

Built with 🐍 Python · 🧠 LangChain · 🔍 ChromaDB · ✨ OpenAI Embeddings · 🖥️ Gradio

</div>
