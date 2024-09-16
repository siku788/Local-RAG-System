# Document Processor and RAG System

This project implements a Streamlit-based web application that processes various document types (PDF, TXT, and web pages) and creates a Retrieval-Augmented Generation (RAG) system for question answering.

## Features

- Support for multiple document types:
  - PDF files
  - Text files
  - Web pages (via URL)
- Flexible embedding model selection:
  - Ollama
  - FastEmbeddings
  - HuggingFace
- Customizable local LLM model selection
- Vector store creation for efficient document retrieval
- RAG-based question answering system
- Streamlit-based user interface for easy interaction
- Comprehensive logging system

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- PyPDF2
- BeautifulSoup4
- Requests
- Ollama (for local LLM serving)
- Other dependencies (see `requirements.txt`)

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/document-processor-rag-system.git
   cd document-processor-rag-system
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama by following the instructions at [Ollama's official website](https://ollama.ai/download).

5. Start the Ollama server:
   ```
   ollama serve
   ```

6. Pull the necessary models. For example, to use the 'mistral' model:
   ```
   ollama pull mistral
   ```

## Usage

1. Ensure the Ollama server is running (step 5 in Setup).

2. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Use the sidebar to configure:
   - Log level
   - Embedding model
   - Local LLM model (enter the name of an Ollama-compatible model, e.g., 'mistral')

5. Upload PDF or TXT files using the file uploader.

6. (Optional) Enter a website URL to process.

7. Click "Process Files/URLs and Create Vector Store" to process the documents and create the RAG system.

8. Once processing is complete, use the "Ask a Question" section to query the system about the processed documents.

## Model Constraints

When selecting a local LLM model, keep in mind the following constraints:

- The model must be compatible with Ollama.
- Ensure you have sufficient system resources (RAM, GPU) to run the chosen model.
- Some larger models may require more processing time, affecting the responsiveness of the application.

To use a specific model:
1. Pull the model using Ollama (e.g., `ollama pull mistral`)
2. Enter the model name in the "Local LLM Model" field in the Streamlit sidebar.

## Project Structure

- `app.py`: Main Streamlit application file
- `uploaded_files/`: Directory for temporarily storing uploaded files
- `logs/`: Directory for storing application logs
- `vector_database/`: Directory for storing the Chroma vector database

## Customization

- To add support for additional document types, modify the `process_file` function in `app.py`.
- To change the embedding models, update the respective sections in the sidebar configuration.
- To use different LLM models, ensure they are pulled via Ollama and enter their names in the sidebar.

## Logging

The application generates detailed logs in the `logs/` directory. Check these logs for debugging and monitoring purposes.

## Troubleshooting

- If you encounter issues with Ollama, ensure the server is running (`ollama serve`).
- If a model is not found, make sure you've pulled it using `ollama pull [model-name]`.
- For performance issues, try using a smaller or more efficient model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details."# Local-RAG-System" 
