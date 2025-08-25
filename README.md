# Agri-RAG: Multimodal RAG System for Agricultural Document Analysis

This project implements a production-ready, multimodal Retrieval-Augmented Generation (RAG) system for analyzing agricultural documents. It processes PDFs to extract both text and images, stores their vector embeddings in a Qdrant Cloud database, and uses an NVIDIA NIM to answer user queries through a web-based interface.

![UI Screenshot](https://github.com/user-attachments/assets/d1843d85-ae3c-4c47-bcc6-14cedc626df8)

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [How It Works](#how-it-works)

## Features

-   **Web-Based Interface**: A user-friendly web UI built with FastAPI and Jinja2 for easy interaction.
-   **Multimodal Data Processing**: Extracts and processes both text and images from PDF files.
-   **Advanced Embeddings**: Uses `sentence-transformers` for text embeddings and OpenAI's `CLIP` for image embeddings.
-   **Vector Database**: Leverages Qdrant Cloud for efficient storage and retrieval of multimodal vectors.
-   **NVIDIA NIM Integration**: Integrates with the NVIDIA NIM API for generating high-quality, context-aware answers.
-   **Chat History**: Saves conversation history, allowing users to review past interactions.

## Tech Stack

-   **Backend**: FastAPI
-   **Frontend**: HTML, CSS, JavaScript, Jinja2
-   **PDF Parsing**: `PyMuPDF` (fitz)
-   **Text Chunking**: `langchain`
-   **Text Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
-   **Image Embeddings**: `transformers` with `OpenAI CLIP` (`openai/clip-vit-base-patch32`)
-   **Vector Database**: `Qdrant Cloud`
-   **LLM**: NVIDIA NIM (`meta/llama-3.1-70b-instruct`)
-   **Programming Language**: Python

## Project Structure

```
.
├── assets/
│   ├── full_logo_white.svg
│   └── green-logo-sm.svg
├── history/
├── pdfs-data/
│   └── (Your PDF files)
├── templates/
│   └── index.html
├── .env
├── .gitignore
├── app.py
├── ingest.py
├── plant_names.json
├── README.md
└── requirements.txt
```

-   `app.py`: The main application script that runs the FastAPI web server.
-   `ingest.py`: The script for processing PDFs, generating embeddings, and populating the Qdrant database.
-   `requirements.txt`: A list of all the Python dependencies required for the project.
-   `.env`: A file to store your secret keys and other environment variables (you need to create this).
-   `pdfs-data/`: The directory where you should place all your PDF files.
-   `templates/index.html`: The HTML template for the web interface.
-   `assets/`: Directory for static assets like logos and images.
-   `history/`: Directory where chat history is stored.
-   `plant_names.json`: A JSON file containing plant names.

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/likhith1409/Agri-RAG.git
    cd Agri-RAG
    ```

2.  **Create the Environment File**:
    Create a file named `.env` in the root of the project directory and add your credentials in the following format:

    ```
    QDRANT_URL="YOUR_QDRANT_URL"
    QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
    NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
    NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
    ```

3.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

The project is divided into two main steps: data ingestion and running the application.

### Step 1: Data Ingestion

Before you can ask questions, you need to process your PDFs and populate the Qdrant database. Place your PDF files in the `pdfs-data` directory and run the ingestion script from your terminal:

```bash
python ingest.py
```

**Note**: This process can take a significant amount of time, depending on the number and size of your PDF files. The script will display progress bars to keep you updated.

### Step 2: Running the Application

Once the data ingestion is complete, you can start the web application. Run the `app.py` script:

```bash
uvicorn app:app --reload
```

The application will be available at `http://127.0.0.1:8000`. Open this URL in your web browser to start asking questions.

## How It Works

1.  **Data Ingestion**: The `ingest.py` script reads PDFs from the `pdfs-data` directory, extracts text and images, and converts them into vector embeddings. These embeddings are then stored in a Qdrant Cloud collection.
2.  **User Query**: The user enters a query into the web interface.
3.  **Multimodal Search**: The application converts the query into a vector and searches the Qdrant collection for the most relevant text and image chunks.
4.  **Contextual Prompting**: The retrieved text and image data are used to construct a detailed prompt for the NVIDIA NIM.
5.  **Answer Generation**: The NIM generates a comprehensive answer based on the provided context.
6.  **Display Results**: The answer, along with source information, is displayed to the user in the web interface.
