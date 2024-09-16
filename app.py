import streamlit as st
import os
from pathlib import Path
import datetime
import re
import logging
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Constants
SAVE_DIR = Path("uploaded_files")
SAVE_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR = Path("vector_database")
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

def setup_logging(log_level):
    """Set up logging configuration."""
    log_file = LOG_DIR / f"app_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def add_metadata(chunks, doc_title):
    """Add metadata to text chunks."""
    return [
        {"text": chunk, "metadata": {
            "title": doc_title,
            "author": "company",
            "date": str(datetime.date.today())
        }} for chunk in chunks
    ]

def sanitize_collection_name(name):
    """Sanitize collection name to meet the requirements."""
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name.ljust(3, 'x')[:63]

def process_file(file_path, logger):
    """Process a single file (PDF, TXT) or website."""
    try:
        logger.info(f"Processing file: {file_path}")
        
        if file_path.startswith("http"):
            loader = WebBaseLoader(file_path)
            documents = loader.load()
            doc_title = file_path
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            doc_title = os.path.basename(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
            documents = loader.load()
            doc_title = os.path.basename(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        if not documents:
            raise ValueError("No documents loaded.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No chunks created.")
        
        metadata_chunks = add_metadata([doc.page_content for doc in chunks], doc_title)
        
        logger.info(f"Successfully processed {file_path}: {len(metadata_chunks)} chunks created")
        return metadata_chunks
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def get_embedding_model(embedding_type, logger):
    """Get the specified embedding model."""
    try:
        if embedding_type == "Ollama":
            return OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
        elif embedding_type == "FastEmbeddings":
            return FastEmbedEmbeddings()
        elif embedding_type == "HuggingFace":
            return HuggingFaceEmbeddings()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    except Exception as e:
        logger.error(f"Error initializing {embedding_type} embedding model: {e}")
        return None

def create_vector_store(processed_documents, embedding_model, logger):
    """Create a vector store from processed documents."""
    try:
        logger.info("Creating vector store")
        all_chunks = []
        for file, chunks in processed_documents.items():
            all_chunks.extend([Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in chunks])
        
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=str(VECTOR_DB_DIR),
            collection_name="document-rag"
        )
        vector_store.persist()
        logger.info(f"Successfully created vector store with {len(all_chunks)} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

def setup_rag_chain(vector_store, local_model, logger):
    """Set up the RAG chain."""
    try:
        logger.info("Setting up RAG chain")
        llm = ChatOllama(model=local_model)
        
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=llm,
            prompt=QUERY_PROMPT
        )

        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("Successfully set up RAG chain")
        return chain
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {e}")
        return None

# Streamlit app
st.title("Document Processor and RAG System")

# Sidebar for configuration
st.sidebar.header("Configuration")
log_level = st.sidebar.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
embedding_type = st.sidebar.selectbox("Embedding Model", ["Ollama", "FastEmbeddings", "HuggingFace"])
local_model = st.sidebar.text_input("Local LLM Model", value="mistral")

# Set up logging
logger = setup_logging(log_level)

# File upload and processing
st.header("Upload Files or Enter URLs")
uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True, type=["pdf", "txt"])
website_url = st.text_input("Enter a website URL (optional)")

if uploaded_files or website_url:
    if st.button("Process Files/URLs and Create Vector Store"):
        progress_bar = st.progress(0)
        processed_documents = {}
        
        # Process uploaded files
        for i, file in enumerate(uploaded_files):
            file_path = SAVE_DIR / file.name
            file_path.write_bytes(file.read())
            
            processed_chunks = process_file(str(file_path), logger)
            
            if processed_chunks:
                processed_documents[str(file_path)] = processed_chunks
                st.write(f"Processed {file.name} successfully.")
            else:
                st.error(f"Failed to process {file.name}.")
            
            progress_bar.progress((i + 1) / (len(uploaded_files) + (1 if website_url else 0)))
        
        # Process website URL
        if website_url:
            processed_chunks = process_file(website_url, logger)
            
            if processed_chunks:
                processed_documents[website_url] = processed_chunks
                st.write(f"Processed {website_url} successfully.")
            else:
                st.error(f"Failed to process {website_url}.")
            
            progress_bar.progress(1.0)
        
        st.session_state.processed_documents = processed_documents
        
        embedding_model = get_embedding_model(embedding_type, logger)
        if embedding_model:
            vector_store = create_vector_store(processed_documents, embedding_model, logger)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.success("Vector store created successfully.")
                
                rag_chain = setup_rag_chain(vector_store, local_model, logger)
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.success("RAG system is ready for queries.")
                else:
                    st.error("Failed to set up RAG chain. Please check the logs.")
            else:
                st.error("Failed to create vector store. Please check the logs.")
        else:
            st.error("Failed to initialize embedding model. Please check the logs.")

# Query interface
if st.session_state.rag_chain:
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if user_query:
        try:
            answer = st.session_state.rag_chain.invoke(user_query)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error processing query: {e}")
            logger.error(f"Error processing query: {e}")
else:
    st.info("Please process files/URLs and create a vector store to enable querying.")

st.sidebar.info("Check the logs directory for detailed process logs.")