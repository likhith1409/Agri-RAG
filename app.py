import os
import io
import re
import base64
import fitz  # PyMuPDF
import json
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Optional, Dict, Any

# Load environment variables
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
COLLECTION_NAME = "multimodal_rag_data"
PDF_DIRECTORY = "pdfs-data"
HISTORY_DIRECTORY = "history"

# Embedding model names
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# LLM Configuration
LLM_MODEL = "meta/llama-3.1-70b-instruct"

# --- Initialize Clients and Models ---
print("Initializing clients and models...")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
text_embedding_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
image_embedding_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
llm_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
print("Initialization complete.")

# --- FastAPI App ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure history directory exists
os.makedirs(HISTORY_DIRECTORY, exist_ok=True)

# Mount the directory containing PDFs as a static path
app.mount("/pdfs-data", StaticFiles(directory="pdfs-data"), name="pdfs-data")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "gemini_api_key": GEMINI_API_KEY})

@app.get("/plant_names.json")
async def get_plant_names():
    return FileResponse('plant_names.json')

@app.get("/history")
async def get_history():
    """Returns a list of all chat history files."""
    files = os.listdir(HISTORY_DIRECTORY)
    # Sort files by creation time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIRECTORY, x)), reverse=True)
    return {"history": files}

@app.get("/history/{chat_id}")
async def get_chat_history(chat_id: str):
    """Returns the content of a specific chat history file."""
    file_path = os.path.join(HISTORY_DIRECTORY, chat_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")
    
    with open(file_path, 'r') as f:
        history_data = json.load(f)
    return history_data

class ChatMessage(BaseModel):
    user: str
    bot: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    history: List[ChatMessage] = Field(default_factory=list)
    chat_id: Optional[str] = None

# --- Search and Generation Functions ---
def search_multimodal(query, top_k=5):
    """Performs a multimodal search in the Qdrant collection."""
    # Get text embedding for the query
    query_text_embedding = text_embedding_model.encode(query, convert_to_tensor=True)
    
    # Get image-compatible embedding for the query
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_image_embedding = image_embedding_model.get_text_features(**inputs)

    # Perform search for text
    text_search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(name="text", vector=query_text_embedding.tolist()),
        limit=top_k,
        with_payload=True,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="text"),
                )
            ]
        )
    )
    
    # Perform search for images
    image_search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(name="image", vector=query_image_embedding[0].tolist()),
        limit=top_k,
        with_payload=True,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="image"),
                )
            ]
        )
    )
    
    return text_search_result, image_search_result

def format_context(text_results):
    """Formats the search results into a context string for the LLM."""
    context = "Relevant text excerpts:\n"
    source_info = []
    for result in text_results:
        source = result.payload['source']
        content = result.payload['content']
        page_num = result.payload.get('page_num', 1)
        context += f"- Source: {source}, Page: {page_num}, Content: {content}\n"
        if source not in [s['source'] for s in source_info]:
            source_info.append({"source": source, "page_num": page_num})
            
    return context, source_info

def generate_short_description(topic: str, context: str):
    """Generates a short description for a topic using the LLM."""
    prompt = f"""
    Based on the following context, provide a very brief, one or two-sentence description of "{topic}".

    Context:
    {context}

    Description:
    """
    try:
        completion = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in summarizing agricultural topics concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150,
        )
        description = completion.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"Error generating short description for {topic}: {e}")
        return ""

def get_image_data(source, page_num):
    """Extracts the first image from a specific PDF page and encodes it in base64."""
    pdf_path = os.path.join(PDF_DIRECTORY, source)
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file not found at {pdf_path}")
        return None

    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num - 1 < len(doc):
            page = doc.load_page(page_num - 1)
            image_list = page.get_images(full=True)
            if image_list:
                img_info = image_list[0]  # Get the first image
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:image/{ext};base64,{base64_image}"
        doc.close()
    except Exception as e:
        print(f"Error processing image from {source} page {page_num}: {e}")
    
    return None

def generate_answer(query: str, context: str, sources: list, history: List[ChatMessage]):
    """Generates an answer using the LLM, includes source links, and maintains conversation history."""
    source_links = [f"[{s['source']}]({PDF_DIRECTORY}/{s['source']})" for s in sources]

    # Construct conversation history for the LLM
    messages = []
    for message in history:
        messages.append({"role": "user", "content": message.user})
        if message.bot and message.bot.get("answer"):
            messages.append({"role": "assistant", "content": message.bot["answer"]})

    # System prompt to define the persona and instructions
    system_prompt = """
    You are a friendly and knowledgeable expert assistant on plant pathology, designed to help agriculture students and professionals.
    Your goal is to provide clear, easy-to-understand answers based on the provided context.
    After answering, ask a follow-up question to encourage conversation and help the user learn more.
    If the context is not sufficient to answer the question, state that you don't have enough information and suggest what you can help with.
    """
    messages.insert(0, {"role": "system", "content": system_prompt})

    # User's current query with context
    user_prompt = f"""
    Use the following context to answer my question.

    Context:
    {context}

    Question: {query}
    """
    messages.append({"role": "user", "content": user_prompt})

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.5,
        top_p=0.8,
        max_tokens=1024,
    )

    answer = completion.choices[0].message.content
    if source_links:
        answer += "\n\n**Sources:**\n- " + "\n- ".join(source_links)
    return answer

# --- API Endpoint ---
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        print(f"Received query: {request.query}")

        print("Searching for relevant information...")
        text_results, _ = search_multimodal(request.query)

        if not text_results:
            return {"answer": "I couldn't find any specific information about that topic. Could you try rephrasing your question, or asking about something else related to plant diseases?", "images": [], "sources": []}

        context, source_info = format_context(text_results)

        print("Generating answer...")
        answer = generate_answer(request.query, context, source_info, request.history)

        # Find topics (list items) in the answer and embed related images
        topic_pattern = re.compile(r"^\s*[-*]\s*(.*)")
        
        final_answer_parts = []
        original_answer_lines = answer.split('\n')

        for line in original_answer_lines:
            final_answer_parts.append(line)
            match = topic_pattern.match(line)
            
            if match:
                topic_clean = match.group(1).strip()
                print(f"Processing topic: {topic_clean}")

                # 1. Search for context for the topic
                topic_text_results, image_results = search_multimodal(topic_clean, top_k=1)

                # 2. Generate a short description
                description_html = ""
                if topic_text_results:
                    topic_context, _ = format_context(topic_text_results)
                    description = generate_short_description(topic_clean, topic_context)
                    if description:
                        description_html = f"<div class='topic-description'>{description}</div>"

                # 3. Get the image
                image_html = ""
                if image_results:
                    payload = image_results[0].payload
                    source = payload.get('source')
                    page_num = payload.get('page_num')
                    if source and page_num:
                        print(f"Found image for '{topic_clean}' in {source} page {page_num}")
                        image_data = get_image_data(source, page_num)
                        if image_data:
                            image_html = f'<img src="{image_data}" alt="{topic_clean}" class="topic-image" onclick="openModal(this.src)">'

                # 4. Combine description and image, and append
                if description_html or image_html:
                    replacement_payload = f"<div class='topic-details'>{image_html}{description_html}</div>"
                    final_answer_parts.append(replacement_payload)
        
        answer = '\n'.join(final_answer_parts)

        # Save history
        updated_history = request.history + [ChatMessage(user=request.query, bot={"answer": answer, "sources": source_info})]
        
        # If it's a new chat, create a new file. Otherwise, overwrite the existing one.
        if not request.chat_id:
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
             history_filename = os.path.join(HISTORY_DIRECTORY, f"chat_{timestamp}.json")
        else:
            history_filename = os.path.join(HISTORY_DIRECTORY, request.chat_id)


        with open(history_filename, 'w') as f:
            json.dump([message.dict() for message in updated_history], f, indent=2)
        print(f"Chat history saved to {history_filename}")

        return {"answer": answer, "images": [], "sources": source_info, "chat_id": os.path.basename(history_filename)}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Application Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
