# 🎓 College Assistant

> An intelligent college Q&A chatbot that routes student queries to the right knowledge source — Academic Handbook, Fee Structure, or General Knowledge — using a conditional LangGraph pipeline with dual FAISS vector stores and Groq Llama 3.3 70B.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Conditional%20RAG-purple?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70B-black?style=flat)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-blue?style=flat)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat&logo=streamlit)](https://collegeassistant-ssiphg98ymdbavfvq68n6p.streamlit.app/)

---

## 🚀 Live Demo

**[👉 Try College Assistant on Streamlit](https://collegeassistant-ssiphg98ymdbavfvq68n6p.streamlit.app/)**

---

## 🔍 What Is This?

**College Assistant** is a programme-aware RAG chatbot for students of IITM (Institute of Innovation in Technology & Management). Ask anything about attendance rules, exam marks, promotion criteria, fee deadlines, scholarships, or late payment charges — and the assistant retrieves the exact answer from the official college PDFs.

The system uses a **LangGraph classifier node** to decide which retriever to use — Academic Handbook RAG, Fee Structure RAG, or direct LLM response — before generating a personalized answer based on the student's selected programme (BCA / BBA / B.Com H).

---

## ✨ Features

- **LLM-based Query Classifier** — Groq Llama 3.3 classifies every query into `academic`, `fee`, or `general` before routing
- **Dual FAISS Vector Stores** — Two separate retrievers: one for the Academic Handbook, one for the Fee Structure
- **Conditional Routing** — LangGraph `add_conditional_edges` routes to the right RAG node based on classifier output
- **Programme-Aware Answers** — Select BCA, BBA, or B.Com(H); responses are personalized to highlight your programme's specific rules and fees
- **Persistent Chat History** — Full multi-turn conversation using `add_messages` and `st.session_state`
- **Query Type Badge** — Each assistant response shows a coloured badge: ACADEMIC 🔵 / FEE 🟡 / GENERAL 🟢
- **Cached Resources** — `@st.cache_resource` ensures PDFs are chunked and FAISS indexes built only once
- **FAISS Index Persistence** — CLI version saves FAISS indexes to disk on first run, reloads on subsequent runs
- **HuggingFace Embeddings** — Uses `all-MiniLM-L6-v2` locally — no embedding API cost

---

## 🏗️ Pipeline Architecture

```
User Query
    ↓
[classifier_node]  — Groq LLM classifies query
    ↓
route_query()
    ├── "academic" → [academic_rag_node]
    │                    ↓
    │              FAISS retriever (academics_handbook.pdf)
    │                    ↓
    ├── "fee"     → [fee_rag_node]
    │                    ↓
    │              FAISS retriever (fee_structure.pdf)
    │                    ↓
    └── "general" → [general_node]
                         ↓
                   NO_RETRIEVAL_NEEDED
                         ↓
                   [response_node]
                   Groq LLM generates answer
                   personalized to student's programme
                         ↓
                      [END]
```

---

## 🧠 State Design

```python
class State(TypedDict):
    programme:         str                          # Student's programme (BCA/BBA/B.Com H)
    messages:          Annotated[list, add_messages] # Full conversation history
    query_type:        str                          # "academic" / "fee" / "general"
    retrieved_context: str                          # Retrieved chunks or NO_RETRIEVAL_NEEDED
```

---

## 🔧 Node Details

### 🔍 `classifier_node`
Sends the user's latest message to Groq LLM with a strict classification prompt. Returns exactly one of: `academic`, `fee`, or `general`.

```
academic → attendance, exams, grading, credits, promotion,
           course structure, summer training, degree requirements
fee      → tuition, payment, refund, late charges, scholarships
general  → greetings, casual talk, anything else
```

### 📘 `academic_rag_node`
Invokes the FAISS retriever built from `academics_handbook.pdf`. Returns top 4 chunks (k=4) as context.

### 💰 `fee_rag_node`
Invokes the FAISS retriever built from `fee_structure.pdf`. Returns top 4 chunks (k=4) as context.

### 💬 `general_node`
Sets `retrieved_context = "NO_RETRIEVAL_NEEDED"` — signals the response node to answer from LLM knowledge directly.

### 📝 `response_node`
Generates the final personalized answer. If context exists, grounds the answer in retrieved document chunks and highlights programme-specific details. If no context, answers from general knowledge.

---

## 🗺️ LangGraph Setup

```python
graph = StateGraph(State)

graph.add_node("classifier",   classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag",      fee_rag_node)
graph.add_node("general",      general_node)
graph.add_node("response",     response_node)

graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", route_query)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag",      "response")
graph.add_edge("general",      "response")

graph.add_edge("response", END)
```

---

## 📄 Knowledge Base

| Document | Content |
|---|---|
| `academics_handbook.pdf` | Attendance policy (75% min), exam marking scheme, grading (CPI), promotion rules, reappear policy, summer training, degree requirements, BCA course structure |
| `fee_structure.pdf` | Programme-wise annual fees (BCA/BBA/B.Com H), payment deadlines, late payment penalties, refund policy, scholarships, education loan support |

---

## 💡 Example Questions to Try

```
What is the minimum attendance required?
```
```
What is the fee for BCA first year?
```
```
What happens if I fail to get promoted?
```
```
What are the late payment charges?
```
```
How many credits do I need to complete BCA?
```
```
Tell me about scholarship options available.
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`, temp=0.4) |
| Workflow | LangGraph `StateGraph` — Conditional Routing |
| Vector Store | FAISS (dual stores) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| PDF Parsing | LangChain `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` (chunk=800, overlap=100) |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
college_assistant/
├── app3.py                   # Streamlit web UI (main entry point)
├── conditional_RAG.py        # CLI version with FAISS index persistence
├── states.py                 # State definitions
├── academics_handbook.pdf    # IITM Academic Handbook 2024-25
├── fee_structure.pdf         # IITM Fee Structure 2024-25
└── requirements.txt
```

---

## 🚀 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/harshit8204/college_assistant.git
cd college_assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit langchain langchain-groq langchain-community langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers pypdf langgraph python-dotenv
```

### 4. Set Up Environment Variables

Create a `.env` file in the root folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at → [console.groq.com](https://console.groq.com)

### 5. Run the App

```bash
streamlit run app3.py
```

Open `http://localhost:8501` in your browser.

### Run CLI Version

```bash
python conditional_RAG.py
```

---

## 📦 Core Dependencies

```
streamlit
langchain
langchain-groq
langchain-community
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
langgraph
python-dotenv
```

---

## 👤 Author

**Harshit Pal** — [@harshit8204](https://github.com/harshit8204)
