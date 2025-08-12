import os
import io
import fitz  # PyMuPDF
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from tqdm import tqdm
import uuid

# Load environment variables
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PDF_DIRECTORY = "pdfs-data"
COLLECTION_NAME = "multimodal_rag_data"

# Embedding model names
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# --- Initialize Clients and Models ---
print("Initializing clients and models...")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
text_embedding_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
image_embedding_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
print("Initialization complete.")

# --- Text and Image Processing Functions ---
def extract_text_and_images_from_pdf(pdf_path):
    """Extracts text and images from a PDF file."""
    doc = fitz.open(pdf_path)
    text_content = ""
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append({"image": image, "page_num": page_num + 1})
    return text_content, images

def get_text_chunks(text):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_text_embeddings(texts):
    """Generates embeddings for a list of texts."""
    return text_embedding_model.encode(texts, convert_to_tensor=True)

def get_image_embeddings(images):
    """Generates embeddings for a list of images."""
    if not images:
        return []
    with torch.no_grad():
        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        image_features = image_embedding_model.get_image_features(**inputs)
    return image_features

# --- Main Ingestion Logic ---
def main():
    """Main function to run the ingestion process."""
    BATCH_SIZE = 16  # Reduced batch size for low-memory environments
    print("Starting data ingestion process...")

    # Check if collection exists, create if not
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
                "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
            },
        )
        print("Collection created.")
        print("Creating payload index for 'type' field...")
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="type",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        print("Payload index created.")
    else:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists with {collection_info.points_count} points.")
        # Ensure the index exists even if the collection was created before
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="type",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            print("Ensured payload index for 'type' field exists.")
        except Exception as e:
            if "already exists" in str(e):
                print("Payload index for 'type' field already exists.")
            else:
                raise e

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        
        try:
            # --- Process one PDF at a time to save memory ---
            text, images = extract_text_and_images_from_pdf(pdf_path)
            
            # Process and upsert text chunks for the current PDF
            if text:
                text_chunks = get_text_chunks(text)
                
                with tqdm(total=len(text_chunks), desc=f"Text - {pdf_file[:20]}...", leave=False) as pbar:
                    for i in range(0, len(text_chunks), BATCH_SIZE):
                        batch_chunks = text_chunks[i:i+BATCH_SIZE]
                        text_embeddings = get_text_embeddings(batch_chunks)
                        
                        points = []
                        for j, chunk in enumerate(batch_chunks):
                            points.append(
                                models.PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"text": text_embeddings[j].tolist()},
                                    payload={"source": pdf_file, "content": chunk, "type": "text"}
                                )
                            )
                        if points:
                            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                        pbar.update(len(batch_chunks))

            # Process and upsert images for the current PDF
            if images:
                with tqdm(total=len(images), desc=f"Images - {pdf_file[:20]}...", leave=False) as pbar:
                    for i in range(0, len(images), BATCH_SIZE):
                        batch_images_data = images[i:i+BATCH_SIZE]
                        batch_images = [item['image'] for item in batch_images_data]
                        
                        image_embeddings = get_image_embeddings(batch_images)
                        
                        points = []
                        for j, img_data in enumerate(batch_images_data):
                            points.append(
                                models.PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector={"image": image_embeddings[j].tolist()},
                                    payload={"source": pdf_file, "page_num": img_data["page_num"], "type": "image"}
                                )
                            )
                        if points:
                            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                        pbar.update(len(batch_images))

        except Exception as e:
            print(f"\nError processing {pdf_file}: {e}")

    print("\nData ingestion complete!")

if __name__ == "__main__":
    main()
