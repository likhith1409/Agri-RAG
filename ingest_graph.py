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
import networkx as nx
from openai import OpenAI
import json


# Load environment variables
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PDF_DIRECTORY = "pdfs-data"
COLLECTION_NAME = "multimodal_graph_rag_data"

# Embedding model names
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# LLM Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
LLM_MODEL = "meta/llama-3.1-70b-instruct"

# --- Initialize Clients and Models ---
print("Initializing clients and models...")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
text_embedding_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
image_embedding_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
llm_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
print("Initialization complete.")

# --- Graph Extraction ---
def get_graph_from_llm(text_chunks, llm_client):
    """Extracts a knowledge graph from text chunks using an LLM."""
    system_prompt = """
    You are a network graph maker who extracts terms and their relations from a given context.
    Your task is to extract the ontology of terms mentioned in the context. These terms should be interconnected and relevant.
    Format your output as a list of json. Each element of the list contains a pair of terms
    and the relation between them, like the following:
    [
        {
            "node_1": "A concept from extracted ontology",
            "node_2": "A related concept from extracted ontology",
            "edge": "relationship between the two concepts"
        }
    ]
    Only answer with a list of JSON objects. Do not provide any other commentary.
    Do not use any quotes other than ".
    """
    
    graph = nx.Graph()
    for chunk in tqdm(text_chunks, desc="Extracting graph from text"):
        try:
            user_prompt = f"Context:\n```{chunk}```"
            
            completion = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            response = completion.choices[0].message.content.strip()
            
            # Extract JSON from markdown code block if present
            if response.startswith("```json"):
                response = response[7:-4].strip()

            result = json.loads(response)
            for item in result:
                graph.add_edge(item['node_1'], item['node_2'], label=item['edge'])
        except Exception as e:
            print(f"Error processing chunk for graph extraction: {e}")
            continue
            
    return graph

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
            
            if text:
                text_chunks = get_text_chunks(text)
                graph = get_graph_from_llm(text_chunks, llm_client)

                # Ingest nodes
                nodes = list(graph.nodes)
                if nodes:
                    with tqdm(total=len(nodes), desc=f"Nodes - {pdf_file[:20]}...", leave=False) as pbar:
                        for i in range(0, len(nodes), BATCH_SIZE):
                            batch_nodes = nodes[i:i+BATCH_SIZE]
                            node_embeddings = get_text_embeddings(batch_nodes)
                            points = []
                            for j, node in enumerate(batch_nodes):
                                points.append(
                                    models.PointStruct(
                                        id=str(uuid.uuid4()),
                                        vector={"text": node_embeddings[j].tolist()},
                                        payload={"source": pdf_file, "content": node, "type": "node"}
                                    )
                                )
                            if points:
                                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                            pbar.update(len(batch_nodes))

                # Ingest edges
                edges = list(graph.edges(data=True))
                if edges:
                    with tqdm(total=len(edges), desc=f"Edges - {pdf_file[:20]}...", leave=False) as pbar:
                        for i in range(0, len(edges), BATCH_SIZE):
                            batch_edges = edges[i:i+BATCH_SIZE]
                            edge_texts = [f"{u} -> {v}: {data['label']}" for u, v, data in batch_edges]
                            edge_embeddings = get_text_embeddings(edge_texts)
                            points = []
                            for j, (u, v, data) in enumerate(batch_edges):
                                points.append(
                                    models.PointStruct(
                                        id=str(uuid.uuid4()),
                                        vector={"text": edge_embeddings[j].tolist()},
                                        payload={
                                            "source": pdf_file,
                                            "u": u,
                                            "v": v,
                                            "label": data['label'],
                                            "type": "edge"
                                        }
                                    )
                                )
                            if points:
                                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                            pbar.update(len(batch_edges))

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
