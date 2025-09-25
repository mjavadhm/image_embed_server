import os
import zipfile
import shutil
import base64
import logging
from io import BytesIO
from typing import List

import gdown
import torch
import chromadb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# --- Configuration & Constants ---
DB_MOUNT_PATH = "/app/product_db"
MODEL_PATH = "/app/product_db/model"
DOWNLOAD_PATH = "/app/product_db/downloads"
COLLECTION_NAME = "image_products_dino"
API_VERSION = "1.1.0"
TOP_K_RESULTS = 3

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State ---
collection = None
processor = None
model = None
device = "cpu"

# --- Pydantic Models ---
class ActivationRequest(BaseModel):
    gdrive_link: str = Field(..., description="Google Drive link to the database ZIP file")

class DownloadRequest(BaseModel):
    gdrive_folder_link: str = Field(..., description="Google Drive link to a folder")

class SearchRequest(BaseModel):
    base64_image: str = Field(..., description="Input image as a base64 string")

class SearchResult(BaseModel):
    id: str
    persian_name: str
    score: float

# --- FastAPI Application ---
app = FastAPI(
    title="Visual Product Search API",
    description="A service for visual product search using the DINOv2 model.",
    version=API_VERSION,
    docs_url="/",
)

# --- Helper Functions ---
def load_model():
    global model, processor
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model path does not exist: {MODEL_PATH}. Model will not be loaded.")
        return
    try:
        logger.info(f"Loading DINOv2 model from local path '{MODEL_PATH}' onto '{device}'...")
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Critical error while loading model: {e}", exc_info=True)

def decode_base64_image(base64_string: str) -> Image.Image:
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def get_image_embedding(image: Image.Image) -> List[float]:
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().tolist()

def download_and_unzip_db_task(gdrive_link: str):
    global collection
    zip_path = "/tmp/db.zip"
    db_content_path = os.path.join(DB_MOUNT_PATH, "db_data")
    try:
        logger.info(f"Starting database download from: {gdrive_link}")
        gdown.download(gdrive_link, zip_path, quiet=False, fuzzy=True)

        if os.path.exists(db_content_path):
            shutil.rmtree(db_content_path)
        os.makedirs(db_content_path, exist_ok=True)

        logger.info(f"Unzipping database to: {db_content_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(db_content_path)

        logger.info("Connecting to the new ChromaDB instance...")
        db_client = chromadb.PersistentClient(path=db_content_path)
        collection = db_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Database activated successfully. Found {collection.count()} items.")
    except Exception as e:
        logger.error(f"Error during database activation process: {e}", exc_info=True)
        collection = None
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

def download_folder_task(gdrive_folder_link: str):
    try:
        logger.info(f"Starting folder download from: {gdrive_folder_link}")
        if os.path.exists(DOWNLOAD_PATH):
            shutil.rmtree(DOWNLOAD_PATH)
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)

        gdown.download_folder(gdrive_folder_link, output=DOWNLOAD_PATH, quiet=False, use_cookies=False)
        logger.info(f"Folder content downloaded successfully to: {DOWNLOAD_PATH}")
    except Exception as e:
        logger.error(f"Error during folder download process: {e}", exc_info=True)

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    load_model()
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

@app.post("/activate-database", status_code=202)
def activate_database(request: ActivationRequest, background_tasks: BackgroundTasks):
    global collection
    collection = None
    logger.info("Received database activation request. Process will run in the background.")
    background_tasks.add_task(download_and_unzip_db_task, request.gdrive_link)
    return {"message": "Database activation process started in the background. Check the status endpoint."}

@app.post("/download-folder", status_code=202)
def download_folder(request: DownloadRequest, background_tasks: BackgroundTasks):
    logger.info("Received folder download request. Process will run in the background.")
    background_tasks.add_task(download_folder_task, request.gdrive_folder_link)
    return {"message": "Folder download process started in the background."}

@app.post("/search/", response_model=List[SearchResult])
def search(request: SearchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Model is not loaded.")
    if collection is None:
        raise HTTPException(status_code=400, detail="Database is not active. Please activate it first.")

    try:
        query_image = decode_base64_image(request.base64_image)
        query_embedding = get_image_embedding(query_image)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K_RESULTS,
            include=["metadatas", "distances"]
        )

        final_results = []
        if results and results.get('ids'):
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            for i in range(len(ids)):
                score = 1 - distances[i]
                final_results.append({
                    "id": ids[i],
                    "persian_name": metadatas[i].get('persian_name', 'N/A'),
                    "score": score
                })
        return final_results
    except Exception as e:
        logger.error(f"Error during search query processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during search processing.")

@app.get("/status")
def get_status():
    db_status = "active" if collection is not None else "inactive"
    model_status = "loaded" if model is not None else "not_loaded"
    db_count = collection.count() if collection else 0
    return {
        "model_status": model_status,
        "database_status": db_status,
        "products_in_db": db_count
    }
