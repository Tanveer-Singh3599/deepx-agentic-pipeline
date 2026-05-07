from typing import List, Dict, Any
from pymongo import MongoClient

MONGO_URI = "mongodb://Tanveer:%40Tanveer12345@ac-rqleiyn-shard-00-00.do3in0d.mongodb.net:27017,ac-rqleiyn-shard-00-01.do3in0d.mongodb.net:27017,ac-rqleiyn-shard-00-02.do3in0d.mongodb.net:27017/?ssl=true&replicaSet=atlas-hszf5p-shard-0&authSource=admin&retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "deep-detector"
COLLECTION_NAME = "ingestion_db"
STAGED_COLLECTION_NAME = "staged_db"
EMBEDDED_COLLECTION_NAME = "Embedded_docs_db"
RESULTS_COLLECTION_NAME = "result_db"

def get_database():
    """
    Establish a connection to the MongoDB cluster and return the database instance.
    """
    client = MongoClient(MONGO_URI)
    return client[DATABASE_NAME]

def store_records_in_mongo(records: List[Dict[str, Any]]) -> List[str]:
    """
    Takes an array of JSON dictionaries (validated from the API) 
    and appends them to the specified MongoDB collection.
    
    Returns a list of the inserted document ObjectIds as strings.
    """
    if not records:
        return []
        
    db = get_database()
    collection = db[COLLECTION_NAME]
    
    # insert_many efficiently appends the batch of documents into the collection in one go
    result = collection.insert_many(records)
    
    # Return string versions of MongoDB ObjectIds
    return [str(inserted_id) for inserted_id in result.inserted_ids]
