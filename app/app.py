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
from langchain_openai import ChatOpenAI  # Add this import for Mistral API
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ... (keep the existing constants and session state initialization)

# ... (keep the existing helper functions)

def get_llm_model(model_type, model_name, api_key, logger):
    """Get the specified LLM model."""
    try:
        if model_type == "Local (Ollama)":
            return ChatOllama(model=model_name)
        elif model_type == "Mistral AI API":
            return ChatOpenAI(model=model_name, openai_api_key=api_key, base_url="https://api.mistral.ai/v1")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error initializing {model_type} model: {e}")
        return None

def setup_rag_chain(vector_store, llm, logger):
    """Set up the RAG chain."""
    try:
        logger.info("Setting up RAG chain")
        
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

# LLM model selection
model_type = st.sidebar.selectbox("LLM Model Type", ["Local (Ollama)", "Mistral AI API"])
if model_type == "Local (Ollama)":
    model_name = st.sidebar.text_input("Local LLM Model", value="mistral")
    api_key = ""
else:
    model_name = st.sidebar.selectbox("Mistral AI Model", ["mistral-tiny", "mistral-small", "mistral-medium"])
    api_key = st.sidebar.text_input("Mistral AI API Key", type="password")

# Set up logging
logger = setup_logging(log_level)

# ... (keep the existing file upload and processing code)

# Query interface
if st.session_state.vector_store:
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if user_query:
        try:
            llm = get_llm_model(model_type, model_name, api_key, logger)
            if llm:
                rag_chain = setup_rag_chain(st.session_state.vector_store, llm, logger)
                if rag_chain:
                    answer = rag_chain.invoke(user_query)
                    st.write("Answer:", answer)
                else:
                    st.error("Failed to set up RAG chain. Please check the logs.")
            else:
                st.error("Failed to initialize LLM model. Please check the logs.")
        except Exception as e:
            st.error(f"Error processing query: {e}")
            logger.error(f"Error processing query: {e}")
else:
    st.info("Please process files/URLs and create a vector store to enable querying.")

st.sidebar.info("Check the logs directory for detailed process logs.")