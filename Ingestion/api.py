from fastapi import FastAPI, HTTPException
from typing import List
from Ingestion.store import store_records_in_mongo
from Schemas.validation_schema import FileUploadRecord

app = FastAPI(title="File Validation API")

@app.post("/upload", status_code=200)
def create_upload_records(records: List[FileUploadRecord]):
    """
    Endpoint that accepts a JSON array of payloads (up to 10) and validates them
    against the FileUploadRecord model.
    """
    # Enforce maximum batch size manually to return a clear 400 error
    if len(records) > 10:
        raise HTTPException(status_code=400, detail="Maximum batch size is 10 records.")
        
    # Convert validated models to standard dictionaries
    # by_alias=True ensures standard camelCase names are passed to Mongo
    # mode='json' guarantees all Pydantic types (like HttpUrl) are safely coerced into primitives (strings) to prevent MongoDB BSON errors
    data_for_mongo = [record.model_dump(mode='json', by_alias=True) for record in records]
    
    # Try saving to MongoDB (commented out until credentials are set in store.py)
    try:
        inserted_ids = store_records_in_mongo(data_for_mongo)
    except Exception as e:
        # If Mongo fails to connect or insert, return a 500 error
        raise HTTPException(status_code=500, detail=f"Database Insertion Error: {str(e)}")
        
    # At this point, ALL records in the list are guaranteed to be successfully parsed and validated
    return {
        "status": 200,
        "inserted_ids": inserted_ids
    }

if __name__ == "__main__":
    import uvicorn
    # Start the local development server when running the file directly
    uvicorn.run("api:app", host="0.0.0.0", port=6969, reload=True)
