import sys
import os
import time
import asyncio
import threading
import uvicorn
import logging
import traceback
import requests
from fastapi import BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorClient
from Agent.hallucination_engine import analyze_document_hallucinations, logger as agent_logger
from Schemas.validation_schema import AnalysisReport
from Ingestion.store import MONGO_URI, DATABASE_NAME, RESULTS_COLLECTION_NAME

# --- Path Configuration ---
# We inject the necessary directories into sys.path to satisfy the local imports 
# of the submodules without needing to alter them.
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

parsing_dir = os.path.join(base_dir, "Parsing")
if parsing_dir not in sys.path:
    sys.path.insert(0, parsing_dir)

ingestion_dir = os.path.join(base_dir, "Ingestion")
if ingestion_dir not in sys.path:
    sys.path.insert(0, ingestion_dir)

# --- Module Imports ---
from Parsing.router import route_unprocessed_records
from Ingestion.api import app

# --- Configuration ---
BACKEND_URL = "https://unstagnant-elida-heartrendingly.ngrok-free.dev/api/v3/analytics/hallucinations/receive"

# --- Result Persistence Logic ---

async def save_analysis_result(payload: dict):
    """
    Saves the final analysis results (SUCCESS or FAILED) to MongoDB (result_db)
    and pushes the result immediately to the backend URL via a simple POST.
    """
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[RESULTS_COLLECTION_NAME]
    
    doc_id = payload.get("document_metadata", {}).get("primary_doc_id")
    if not doc_id:
        agent_logger.error("Attempted to save result without a doc_id.")
        return
        
    payload["is_delivered"] = False
    payload["created_at"] = time.time()
        
    # 1. Update Database (result_db)
    await collection.update_one(
        {"document_metadata.primary_doc_id": doc_id},
        {"$set": payload},
        upsert=True
    )
    agent_logger.info(f"Results archived to result_db for {doc_id}.")

    # 2. Simple POST Push: Immediate delivery to Backend (using requests)
    try:
        agent_logger.info(f"Sending simple POST to {BACKEND_URL}...")
        # Running in a thread to prevent blocking the async loop
        await asyncio.to_thread(requests.post, BACKEND_URL, json=payload, timeout=10.0)
        agent_logger.info(f"Report successfully pushed for {doc_id}.")
    except Exception as e:
        agent_logger.error(f"Failed to push POST for {doc_id}: {str(e)}")

async def run_hallucination_analysis_task(doc_id: str, text: str):
    """
    Fire-and-Forget task: Runs LangGraph analysis and persists results for polling.
    """
    try:
        # 1. Run the Agentic Hallucination Analysis
        result_json = await analyze_document_hallucinations(doc_id, text)
        
        # 2. Persist to MongoDB (Polling Model)
        await save_analysis_result(result_json)
        
    except Exception as e:
        agent_logger.error(f"System-level crash in background task for doc_id {doc_id}.")
        failure_payload = {
            "document_metadata": {
                "primary_doc_id": doc_id,
                "status": "FAILED",
                "analysis_timestamp": time.ctime()
            },
            "error_code": "BACKGROUND_TASK_CRASH",
            "error_message": f"Critical system error: {str(e)}"
        }
        await save_analysis_result(failure_payload)



@app.post("/get-hallucination-results")
async def get_hallucination_results_endpoint():
    """
    Result Retrieval Endpoint: Returns the oldest undelivered results from result_db.
    Converts 'get statement to post statement' as requested.
    """
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[RESULTS_COLLECTION_NAME]
    
    # 1. Find undelivered results, sorted by creation time (FIFO)
    cursor = collection.find({"is_delivered": False}).sort("created_at", 1).limit(10)
    results = await cursor.to_list(length=10)
    
    if not results:
        return []
        
    # 2. Extract IDs for atomic update
    record_ids = [r["document_metadata"]["primary_doc_id"] for r in results]
    
    # 3. Mark as delivered so they are consumed once
    await collection.update_many(
        {"document_metadata.primary_doc_id": {"$in": record_ids}},
        {"$set": {"is_delivered": True}}
    )
    
    # Clean up internal MongoDB IDs
    for r in results:
        r.pop("_id", None)
        r.pop("is_delivered", None)
        
    agent_logger.info(f"Delivered {len(results)} analysis reports via POST polling.")
    return results




async def run_parser_loop():
    """
    Background loop that continuously queries the 'queued' records
    and processes them using functions found in the Parsing directory.
    Now automatically triggers Hallucination Analysis on success.
    """
    print("Starting background parser loop (Automated Analysis Mode)...")
    while True:
        try:
            # Capturing the list of successfully processed documents
            processed_docs = route_unprocessed_records()
            
            for doc_id, text in processed_docs:
                print(f"Auto-triggering hallucination analysis for {doc_id}...")
                # Schedule the async analysis task without blocking the loop
                asyncio.create_task(run_hallucination_analysis_task(doc_id, text))
                
        except Exception as e:
            print(f"Parser encountered an error in the loop: {e}")
        
        # Poll every 5 seconds (async sleep)
        await asyncio.sleep(5)

def start_parser_thread():
    """
    Bridge to run the async parser loop in a dedicated daemon thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_parser_loop())

if __name__ == "__main__":
    print("Starting Pipeline... (Parser Loop + FastAPI Ingestion)")
    
    # 1. Initialize the Parser loop in a background daemon thread
    parser_thread = threading.Thread(target=start_parser_thread, daemon=True)
    parser_thread.start()
    
    # 2. Run the Ingestion FastAPI service on the main thread
    uvicorn.run(app, host="0.0.0.0", port=6969)
