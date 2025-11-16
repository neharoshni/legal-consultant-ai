"""
app.py - Main Legal Consultant Chatbot Application
Complete working system with Groq LLM integration
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import streamlit as st
from groq import Groq
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Best free Groq model
    EMBEDDING_MODEL = "law-ai/InLegalBERT"
    DATA_FOLDER = "legal_data"  # Folder containing your legal documents
    VECTOR_DB_PATH = "vector_db"
    MAX_RETRIEVAL = 5
    CHUNK_SIZE = 512
    
# ============================================================================
# EMBEDDER - InLegalBERT
# ============================================================================

class InLegalBERTEmbedder:
    """Generate embeddings using InLegalBERT"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class LegalDocumentLoader:
    """Load legal documents from data folder"""
    
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
    
    def load_all_documents(self) -> List[Dict]:
        """Load all documents from data folder"""
        documents = []
        
        # Load JSON files
        for json_file in self.data_folder.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
        
        # Load TXT files
        for txt_file in self.data_folder.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                doc = {
                    'text': text,
                    'metadata': {
                        'source': txt_file.stem,
                        'type': 'text_file'
                    }
                }
                documents.append(doc)
        
        print(f"✓ Loaded {len(documents)} documents from {self.data_folder}")
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks"""
        chunked = []
        
        for doc in documents:
            text = doc['text']
            metadata = doc.get('metadata', {})
            
            # Simple chunking by character count
            chunks = []
            for i in range(0, len(text), Config.CHUNK_SIZE):
                chunk = text[i:i + Config.CHUNK_SIZE]
                if len(chunk) > 50:  # Skip very small chunks
                    chunks.append(chunk)
            
            # Create document for each chunk
            for idx, chunk in enumerate(chunks):
                chunked_doc = {
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_id': idx,
                        'total_chunks': len(chunks)
                    }
                }
                chunked.append(chunked_doc)
        
        print(f"✓ Created {len(chunked)} chunks from {len(documents)} documents")
        return chunked

# ============================================================================
# RAG SYSTEM
# ============================================================================

class LegalRAGSystem:
    """RAG system with FAISS vector database"""
    
    def __init__(self, embedder: InLegalBERTEmbedder):
        self.embedder = embedder
        self.documents = []
        self.index = None
        self.dimension = 768
        self.db_path = Path(Config.VECTOR_DB_PATH)
        self.db_path.mkdir(exist_ok=True)
    
    def build_index(self, documents: List[Dict]):
        """Build FAISS index from documents"""
        print("Building vector index...")
        
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"✓ Index built with {len(documents)} documents")
        
        # Save index and documents
        self.save()
    
    def save(self):
        """Save index and documents to disk"""
        if self.index:
            faiss.write_index(self.index, str(self.db_path / "index.faiss"))
            
            with open(self.db_path / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Saved index to {self.db_path}")
    
    def load(self):
        """Load index and documents from disk"""
        index_path = self.db_path / "index.faiss"
        docs_path = self.db_path / "documents.json"
        
        if index_path.exists() and docs_path.exists():
            self.index = faiss.read_index(str(index_path))
            
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            print(f"✓ Loaded index with {len(self.documents)} documents")
            return True
        
        return False
    
    def retrieve(self, query: str, top_k: int = Config.MAX_RETRIEVAL) -> List[Tuple[Dict, float]]:
        """Retrieve most relevant documents"""
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results

# ============================================================================
# GROQ LLM INTEGRATION
# ============================================================================

class GroqLLM:
    """Groq API integration for LLM"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = Config.GROQ_MODEL
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using Groq"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Indian legal consultant AI. Provide accurate, helpful legal advice based on Indian law. Always cite relevant sections and acts. Be clear about when users should consult a lawyer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ============================================================================
# LEGAL CHATBOT
# ============================================================================

class LegalChatbot:
    """Main chatbot interface"""
    
    def __init__(self, rag_system: LegalRAGSystem, llm: GroqLLM):
        self.rag_system = rag_system
        self.llm = llm
    
    def create_prompt(self, query: str, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        """Create prompt with retrieved context"""
        
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            section = metadata.get('section', '')
            
            context_parts.append(
                f"[Reference {i}] (Relevance: {score:.2f})\n"
                f"Source: {source} {section}\n"
                f"Content: {doc['text'][:500]}...\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following legal references, provide accurate legal advice for the user's query.

LEGAL REFERENCES:
{context}

USER QUERY:
{query}

INSTRUCTIONS:
1. Analyze the query and relevant legal provisions
2. Cite specific sections, acts, or case laws from the references
3. Provide step-by-step guidance on how to proceed
4. Mention important deadlines, procedures, or requirements
5. Clearly state when professional legal counsel should be consulted
6. Be specific about jurisdiction (Indian law) and any limitations

Provide a clear, actionable response:"""
        
        return prompt
    
    def chat(self, user_query: str) -> Dict:
        """Process user query and generate response"""
        
        # Retrieve relevant documents
        retrieved_docs = self.rag_system.retrieve(user_query)
        
        if not retrieved_docs:
            return {
                'response': "I couldn't find relevant legal information for your query. Please try rephrasing or provide more details.",
                'sources': [],
                'query': user_query
            }
        
        # Create prompt with context
        prompt = self.create_prompt(user_query, retrieved_docs)
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Format sources
        sources = []
        for doc, score in retrieved_docs:
            metadata = doc.get('metadata', {})
            sources.append({
                'text': doc['text'][:200] + "...",
                'source': metadata.get('source', 'Unknown'),
                'section': metadata.get('section', ''),
                'relevance': f"{score:.2f}"
            })
        
        return {
            'response': response,
            'sources': sources,
            'query': user_query,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session_state():
    """Initialize session state"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def initialize_system(groq_api_key: str, rebuild_index: bool = False):
    """Initialize the chatbot system"""
    with st.spinner("Initializing Legal Chatbot System..."):
        
        # Initialize embedder
        st.info("Loading InLegalBERT model...")
        embedder = InLegalBERTEmbedder()
        
        # Initialize RAG system
        st.info("Setting up RAG system...")
        rag_system = LegalRAGSystem(embedder)
        
        # Load or build index
        if not rebuild_index and rag_system.load():
            st.success("✓ Loaded existing vector database")
        else:
            st.info("Building new vector database from legal documents...")
            loader = LegalDocumentLoader(Config.DATA_FOLDER)
            documents = loader.load_all_documents()
            
            if not documents:
                st.error(f"No documents found in {Config.DATA_FOLDER}/")
                return None
            
            chunked_docs = loader.chunk_documents(documents)
            rag_system.build_index(chunked_docs)
            st.success(f"✓ Built index with {len(chunked_docs)} document chunks")
        
        # Initialize LLM
        st.info("Connecting to Groq API...")
        llm = GroqLLM(groq_api_key)
        
        # Create chatbot
        chatbot = LegalChatbot(rag_system, llm)
        st.success("✓ Legal Chatbot Ready!")
        
        return chatbot

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Legal Consultant AI",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Consultant AI Chatbot")
    st.markdown("*AI-powered legal advice based on Indian law using InLegalBERT + RAG*")
    
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your free API key from https://console.groq.com"
        )
        
        st.markdown("---")
        
        rebuild = st.checkbox("Rebuild Vector Database", value=False)
        
        if st.button("Initialize System", type="primary"):
            if not groq_api_key:
                st.error("Please enter your Groq API key")
            else:
                chatbot = initialize_system(groq_api_key, rebuild_index=rebuild)
                if chatbot:
                    st.session_state.chatbot = chatbot
                    st.session_state.initialized = True
        
        st.markdown("---")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### About
        This chatbot uses:
        - **InLegalBERT** for embeddings
        - **FAISS** for vector search
        - **Groq Llama 3.1** for generation
        - **RAG** for up-to-date legal info
        
        ### Data Location
        Legal documents: `legal_data/`
        
        ### Disclaimer
        This AI provides general legal information.
        For specific cases, consult a lawyer.
        """)
    
    # Main chat interface
    if not st.session_state.initialized:
        st.info("👈 Please configure and initialize the system using the sidebar")
        
        st.markdown("### 📁 Setup Instructions")
        st.markdown("""
        1. Create a `legal_data` folder in your project directory
        2. Add your legal documents (JSON or TXT format)
        3. Enter your Groq API key in the sidebar
        4. Click "Initialize System"
        """)
        
        st.markdown("### 📄 Sample Document Format (JSON)")
        st.code('''
[
    {
        "text": "Section 420 IPC: Whoever cheats and thereby dishonestly induces...",
        "metadata": {
            "source": "IPC",
            "section": "420",
            "category": "criminal"
        }
    }
]
        ''', language='json')
    
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message['query'])
            
            with st.chat_message("assistant"):
                st.write(message['response'])
                
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message['sources'], 1):
                        st.markdown(f"**{i}. {source['source']} {source['section']}** (Relevance: {source['relevance']})")
                        st.text(source['text'])
                        st.markdown("---")
        
        # Chat input
        user_query = st.chat_input("Ask your legal question...")
        
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Researching legal information..."):
                    result = st.session_state.chatbot.chat(user_query)
                
                st.write(result['response'])
                
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(f"**{i}. {source['source']} {source['section']}** (Relevance: {source['relevance']})")
                        st.text(source['text'])
                        st.markdown("---")
            
            # Add to history
            st.session_state.chat_history.append(result)

if __name__ == "__main__":
    main()
