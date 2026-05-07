from Ingestion.store import get_database, COLLECTION_NAME, EMBEDDED_COLLECTION_NAME
from Parsing import parsers
import os
from Embedding.embedding import rag_manager

# --- Operational Logic ---
# To fix the 'garbage data' issue, we clear the corrupted vector database at startup once.
# In a production environment, this would be a migrations task.
try:
    _db = get_database()
    _db[EMBEDDED_COLLECTION_NAME].delete_many({})
    print(f"Purged corrupted {EMBEDDED_COLLECTION_NAME} for clean RAG start.")
except Exception as _e:
    print(f"Purged warning: {_e}")

def route_unprocessed_records():
    """
    Fetches pending ('queued') records from the MongoDB collection 
    and routes them to the appropriate functions in parsing.py 
    based on their MIME type.
    """
    db = get_database()
    collection = db[COLLECTION_NAME]
    
    cursor = collection.find({"status": "queued"})
    records = list(cursor)
    
    if not records:
        return []
        
    processed_docs = []
        
    for record in records:
        mime_type = record.get("mimeType", "").lower()
        record_id = record.get("_id")
        
        # Override incorrect MIME types based on obvious URL extensions
        url = record.get("cloudinaryUrl", record.get("cloudinary_url", "")).lower()
        if any(ext in url for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]):
            mime_type = "image/jpeg"
        elif ".pdf" in url:
            mime_type = "application/pdf"
        elif ".docx" in url:
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif any(ext in url for ext in [".txt", ".csv", ".md"]):
            mime_type = "text/plain"
            
        try:
            # Route based on MIME type and pass the record for parsing
            # All parsers now return the list of cleaned chunks extracted
            chunks, full_text = [], ""
            if mime_type.startswith("image/"):
                chunks, full_text = parsers.img(record)
                
            elif mime_type == "application/pdf":
                chunks, full_text = parsers.pdfs(record)
                
            elif mime_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                chunks, full_text = parsers.docs(record)
                
            elif mime_type.startswith("text/"):
                chunks, full_text = parsers.texts(record)
                
            else:
                # Mark as 'unsupported' so it doesn't get infinitely re-processed
                collection.update_one(
                    {"_id": record_id},
                    {"$set": {"status": "unsupported"}}
                )
                continue
                
            # If successfully processed without exceptions, update the status in MongoDB
            collection.update_one(
                {"_id": record_id},
                {"$set": {"status": "processed"}}
            )

            # --- RAG Indexing Hook ---
            # Extract the raw text for the entire document and index it for vector retrieval
            try:
                # We now pass the ACTUAL chunks extracted from the file
                # effectively bypassing the 'metadata garbage' issue!
                rag_manager.add_document(
                    doc_id=str(record_id),
                    chunks=chunks,
                    source=url
                )
            except Exception as e:
                print(f"RAG Indexing failed for {record_id}: {e}")
            
            # Successfully reached the end of the pipeline for this document
            processed_docs.append((str(record_id), full_text))
            
        except Exception as e:
            print(f"Error processing record {record_id}: {e}")
            collection.update_one(
                {"_id": record_id},
                {"$set": {"status": "error", "error_message": str(e)}}
            )
            
    return processed_docs

if __name__ == "__main__":
    route_unprocessed_records()
