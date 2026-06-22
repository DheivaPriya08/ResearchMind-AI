# 📚 ResearchMind AI

### AI-Powered Multi-Paper Research Intelligence Platform

ResearchMind AI is an intelligent research paper analysis platform that helps researchers, students, and professionals extract meaningful insights from multiple research papers using Semantic Search, Automated Summarization, Comparative Analysis, Literature Review Generation, Citation Extraction, and Research Gap Discovery.

🌐 **Live Demo:**  
https://researchmind-ai-dheivapriya.streamlit.app/

---

## 🚀 Features

### 🔍 Semantic Search
- Search across multiple uploaded research papers
- FAISS-powered semantic retrieval
- Relevance score ranking
- Multi-document knowledge discovery

### 📄 Automated Paper Summaries
Generate structured summaries containing:
- Summary
- Methodology
- Results
- Conclusion
- Limitations
- Citation References

### ⚖️ Research Paper Comparison
Compare papers side-by-side based on:
- Dataset
- Model Used
- Accuracy
- Methodology
- Key Findings

### 📚 Literature Review Generation
Automatically generate a consolidated literature review from multiple research papers.

### 🎯 Research Gap Discovery
Identify:
- Research limitations
- Open challenges
- Future research opportunities
- Cross-paper knowledge gaps

### 📥 PDF Report Export
Export generated research insights and analysis reports.

### 🤖 AI-Powered Research Analysis
- Google Gemini (Primary LLM)
- OpenAI Support (Optional)

---

## 🏗️ System Architecture

```text
Research Papers (PDFs)
          │
          ▼
   PDF Text Extraction
          │
          ▼
      Text Chunking
          │
          ▼
 Sentence Transformers
          │
          ▼
 Vector Embeddings
          │
          ▼
      FAISS Index
          │
          ▼
 Semantic Retrieval
          │
          ▼
 Research Intelligence Engine
          │
          ├── Semantic Search
          ├── Paper Summaries
          ├── Citation Extraction
          ├── Paper Comparison
          ├── Literature Review
          ├── Research Gap Discovery
          └── PDF Export
```

---

## 📊 Key Capabilities

✅ Multi-PDF Research Paper Analysis

✅ Semantic Search with Relevance Scores

✅ Automated Research Summarization

✅ Citation Extraction

✅ Comparative Research Evaluation

✅ Literature Review Generation

✅ Research Gap Identification

✅ PDF Report Export

✅ Streamlit Cloud Deployment

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### AI & NLP
- Google Gemini
- Sentence Transformers
- OpenAI (Optional)

### Vector Search
- FAISS

### Document Processing
- pdfplumber
- PyPDF (Fallback Extraction)
- Regular Expressions

### Reporting
- ReportLab

### Programming Language
- Python

---

## 📷 Screenshots

### 🔍 Semantic Search

Search relevant information across multiple research papers with relevance scores.
<img width="1919" height="883" alt="Screenshot (217)" src="https://github.com/user-attachments/assets/81ab32e9-2cf1-42de-87cc-acb6f72b7779" />


---

### 📄 Paper Summaries

Generate structured summaries including methodology, results, conclusions, limitations, and citations.

<img width="1920" height="867" alt="Screenshot (218)" src="https://github.com/user-attachments/assets/0444fc5e-c5f5-4361-8dd3-a5c7d347cf2b" />


---

### ⚖️ Compare Papers

Compare datasets, models, accuracy, and methodologies across papers.

<img width="1920" height="877" alt="image" src="https://github.com/user-attachments/assets/6030abc7-791f-438c-b697-af8aec66c08e" />


---

### 📚 Literature Review & Research Gap Discovery

Generate literature reviews and identify future research opportunities.

<img width="1920" height="870" alt="image" src="https://github.com/user-attachments/assets/e1237718-ef75-4dbb-9e8a-b9622bba0c5a" />


---

## 📂 Project Structure

```text
ResearchMind-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── semantic_search.png
│   ├── summaries.png
│   ├── compare_papers.png
│   └── literature_review.png
│
├── data/
├── reports/
└── assets/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/DheivaPriya08/ResearchMind-AI.git

cd ResearchMind-AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🌐 Live Deployment

Streamlit Cloud:

https://researchmind-ai-dheivapriya.streamlit.app/

---

## 🎯 Use Cases

### Academic Research
- Literature review automation
- Research gap analysis
- Comparative paper analysis

### Students
- Faster paper understanding
- Citation exploration
- Research topic discovery

### Researchers
- Multi-paper knowledge extraction
- Methodology comparison
- Trend identification

### Organizations
- Research intelligence
- Technical document exploration
- Knowledge management

---

## 🔮 Future Enhancements

- Conversational RAG Chat
- Knowledge Graph Visualization
- Citation Network Analysis
- Research Trend Analytics
- Multi-Language Paper Support
- Advanced PDF Annotation
- Research Recommendation Engine

---

## 📈 Project Highlights

- Built an end-to-end AI-powered document intelligence platform
- Implemented Semantic Search using FAISS and Sentence Transformers
- Automated research paper summarization and comparison
- Generated Literature Reviews using LLMs
- Identified Research Gaps across multiple papers
- Extracted Citation Information automatically
- Deployed publicly using Streamlit Cloud

---

## 👩‍💻 Author

### Dheiva Priya

M.Sc. Data Science

GitHub:
https://github.com/DheivaPriya08

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the project

📢 Share with researchers and students

---

## 📜 License

This project is intended for educational and research purposes.

MIT License
