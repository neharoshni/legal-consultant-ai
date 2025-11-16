# 🚀 Legal Consultant AI Chatbot - Implementation Guide

## ✅ Setup Complete!

Your Legal Consultant AI Chatbot is ready to run! Here's what's been set up:

### 📦 Installed Components
- ✅ Python virtual environment (`venv/`)
- ✅ All dependencies installed:
  - Streamlit (Web UI)
  - PyTorch (Deep Learning)
  - Transformers (InLegalBERT)
  - Groq (LLM API)
  - FAISS (Vector Database)
  - NumPy, SentencePiece, Protobuf

### 📁 Project Structure
```
legalModel/
├── legal_bot_complete.py      # Main Streamlit application
├── legal_chatbot_rag.py        # Core RAG implementation
├── legal_data/                 # Legal documents (15 documents loaded)
│   └── sample_legal_data.json
├── vector_db/                  # FAISS index (will be created on first run)
├── venv/                       # Python virtual environment
├── requirements_file.txt       # Dependencies
└── key.txt                     # Groq API key
```

## 🎯 How to Run the Application

### Step 1: Activate Virtual Environment
```powershell
cd legalModel
.\venv\Scripts\activate
```

### Step 2: Run the Streamlit App
```powershell
streamlit run legal_bot_complete.py
```

### Step 3: Configure in Browser
1. The app will open in your browser at `http://localhost:8501`
3. Click "Initialize System"
4. Wait for InLegalBERT model to download and vector database to build
5. Start asking legal questions!

## 💡 Example Questions to Try

1. **Criminal Law**: "I was cheated by a seller who took money but didn't deliver the product. What legal action can I take?"
2. **Consumer Rights**: "How do I file a consumer complaint and what is the jurisdiction?"
3. **Cyber Crime**: "Someone is impersonating me online. What are my legal options?"
4. **Family Law**: "What are the grounds for divorce under Hindu Marriage Act?"
5. **Labor Law**: "What are my rights if I'm being retrenched from my job?"

## 🔧 System Features

### RAG (Retrieval-Augmented Generation)
- Uses **InLegalBERT** for domain-specific legal embeddings
- **FAISS** vector database for fast similarity search
- Retrieves top 5 most relevant legal provisions
- **Groq Llama 3.3** generates contextual responses

### Legal Knowledge Base (15 Documents)
- Indian Penal Code (IPC) sections
- Consumer Protection Act
- IT Act (Cyber crimes)
- Hindu Marriage Act
- Industrial Disputes Act
- Negotiable Instruments Act
- Criminal Procedure Code (CrPC)
- Right to Information Act

## 📊 First Run Process

When you initialize the system for the first time:

1. **InLegalBERT Download** (~300MB) - Takes 2-5 minutes
2. **Document Processing** - Chunks 15 legal documents
3. **Embedding Generation** - Creates vector embeddings
4. **FAISS Index Building** - Builds searchable index
5. **Index Saved** - Stored in `vector_db/` for future use

Subsequent runs will load the saved index (much faster!).

## 🎨 UI Features

- **Chat Interface**: Natural conversation with the AI
- **Source Citations**: View which legal provisions were used
- **Relevance Scores**: See how relevant each source is
- **Chat History**: Review previous questions and answers
- **Rebuild Option**: Rebuild vector database if you add new documents

## 📝 Adding More Legal Documents

To expand the knowledge base:

1. Add JSON or TXT files to `legal_data/` folder
2. JSON format:
```json
[
    {
        "text": "Section X: Legal provision text...",
        "metadata": {
            "source": "Act Name",
            "section": "X",
            "category": "criminal/civil/etc",
            "type": "bare_act/procedure/judgment"
        }
    }
]
```
3. Check "Rebuild Vector Database" in sidebar
4. Click "Initialize System"

## 🌐 Data Sources for Expansion

- **Indian Kanoon** (indiankanoon.org) - 10M+ judgments
- **India Code** (indiacode.nic.in) - All central acts
- **Supreme Court website** - Recent cases
- **Legal blogs** - Explanations and guides

## ⚠️ Important Notes

### Disclaimer
This AI provides **general legal information only**. For specific cases, always consult a qualified lawyer.

### API Key
- Free Groq API key included in `key.txt`
- Get your own at: https://console.groq.com
- Free tier: 30 requests/minute

### Performance
- First run: 5-10 minutes (model download + indexing)
- Subsequent runs: 30-60 seconds (load saved index)
- Query response: 2-5 seconds

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure virtual environment is activated
```powershell
.\venv\Scripts\activate
```

### Issue: "CUDA not available"
**Solution**: Normal! The app uses CPU (faiss-cpu). GPU not required.

### Issue: "Groq API error"
**Solution**: Check API key is correct and you have internet connection

### Issue: "Port already in use"
**Solution**: Run on different port
```powershell
streamlit run legal_bot_complete.py --server.port 8502
```

## 🚀 Next Steps

1. **Run the application** and test with sample questions
2. **Add more legal documents** to expand knowledge base
3. **Customize prompts** in `legal_bot_complete.py` for specific use cases
4. **Deploy** using Streamlit Cloud or Docker

## 📞 Support

For issues or questions:
- Check the code comments in `legal_bot_complete.py`
- Review Streamlit docs: https://docs.streamlit.io
- Groq API docs: https://console.groq.com/docs

---

**Ready to start? Run the command below!**

```powershell
streamlit run legal_bot_complete.py
```

