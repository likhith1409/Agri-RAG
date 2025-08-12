# Multimodal RAG System with NVIDIA LLM and Qdrant

This project implements a production-ready, multimodal Retrieval-Augmented Generation (RAG) system using Python. It processes PDF documents, extracts both text and images, and stores their vector embeddings in a Qdrant Cloud database. The system then uses an NVIDIA LLM to answer questions based on the retrieved multimodal context.

## Features

- **Multimodal Data Processing**: Extracts and processes both text and images from PDF files.
- **Advanced Embeddings**: Uses `sentence-transformers` for text embeddings and OpenAI's `CLIP` for image embeddings.
- **Vector Database**: Leverages Qdrant Cloud for efficient storage and retrieval of multimodal vectors.
- **Orchestration**: Manages the RAG pipeline for retrieving context.
- **Large Language Model**: Integrates with NVIDIA's LLM API for generating high-quality answers.
- **Production-Ready**: Organized into separate scripts for data ingestion and application logic, with clear instructions.

## Tech Stack

- **PDF Parsing**: `PyMuPDF` (fitz)
- **Text Chunking**: `langchain`
- **Text Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Image Embeddings**: `transformers` with `OpenAI CLIP` (`openai/clip-vit-base-patch32`)
- **Vector Database**: `Qdrant Cloud`
- **LLM**: NVIDIA LLM API (`meta/llama-3.1-70b-instruct`)
- **Programming Language**: Python

## Setup and Installation

1.  **Clone the Repository (or download the files)**:
    Make sure you have all the project files (`ingest.py`, `app.py`, `requirements.txt`) in a single directory.

2.  **Create the Environment File**:
    Create a file named `.env` in the root of the project directory and add your credentials in the following format:

    ```
    QDRANT_URL="YOUR_QDRANT_URL"
    QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
    NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
    NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
    ```

3.  **Install Dependencies**:
    Open your terminal or command prompt, navigate to the project directory, and run the following command to install the required Python packages:
    ```bash
    python -m pip install -r requirements.txt
    ```

## Usage

The project is divided into two main steps: data ingestion and running the application.

### Step 1: Data Ingestion

Before you can ask questions, you need to process your PDFs and populate the Qdrant database. Run the ingestion script from your terminal:

```bash
python ingest.py
```

**Note**: This process can take a significant amount of time, depending on the number and size of your PDF files. The script will display progress bars to keep you updated.

### Step 2: Running the Application

Once the data ingestion is complete, you can start the question-answering application. Run the `app.py` script:

```bash
python app.py
```

The application will initialize the models and then prompt you to ask a question. Type your question and press Enter. To exit the application, type `exit`.

## Project Structure

-   `ingest.py`: The script for processing PDFs, generating embeddings, and populating the Qdrant database.
-   `app.py`: The main application script that handles user queries, retrieves context, and generates answers.
-   `requirements.txt`: A list of all the Python dependencies required for the project.
-   `.env`: A file to store your secret keys and other environment variables (you need to create this).
-   `pdfs-data/`: The directory where you should place all your PDF files.
