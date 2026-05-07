import numpy as np
import time
import hashlib
from sentence_transformers import SentenceTransformer
from Ingestion.store import get_database, EMBEDDED_COLLECTION_NAME, STAGED_COLLECTION_NAME
from .processing import clean_text, chunk_text, is_meaningful_legal_content, has_legal_domain_anchors
from .external_retrieval import search_legal_context

# --- Constants & Configuration ---
MODEL_NAME = "intfloat/e5-base"  # Lightweight & 8GB RAM Friendly
SIMILARITY_THRESHOLD = 0.75  # Minimum cosine similarity for a match

class LocalRAGManager:
    """
    A stable, single-instance Vector Retrieval engine for the Hackathon.
    Handles memory-mapped numpy search to avoid C++ compiler issues with FAISS.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalRAGManager, cls).__new__(cls)
            cls._instance.model = SentenceTransformer(MODEL_NAME, device="cpu")
            cls._instance.index = None  # (Matrix of embeddings)
            cls._instance.metadata = [] # (List of dicts matching index)
            cls._instance.load_index()
        return cls._instance

    def load_index(self) -> None:
        """Loads all existing embeddings from Embedded_docs_db at startup."""
        print(f"Loading knowledge base from {EMBEDDED_COLLECTION_NAME}...")
        db = get_database()
        collection = db[EMBEDDED_COLLECTION_NAME]
        
        cursor = collection.find({})
        all_records = list(cursor)
        
        if not all_records:
            print("Knowledge base is currently empty. Ready for document ingestion.")
            return

        self.metadata = all_records
        # Convert all stored embedding lists into a single float32 numpy matrix
        # Note: We now consistently use the 'embedding' key (renamed from 'vector')
        self.index = np.array([r.get("embedding", r.get("vector")) for r in all_records], dtype=np.float32)
        print(f"Index loaded with {len(self.metadata)} chunks.")

    def add_document(self, doc_id: str, chunks: list, source: str = "User"):
        """Process, embed, and store document chunks in both MongoDB and local index."""
        if not chunks:
            return
            
        print(f"Ingesting doc_id {doc_id} into retrieval index ({len(chunks)} chunks)...")
        
        # 1. NEW: Smart Legal Filter & Quality Check
        # First, clean OCR boilerplate from text
        for c in chunks:
            c["text"] = clean_text(c.get("text", ""))
            
        # Second, filter for meaningful semantic content AND domain relevance
        valid_chunks = []
        skipped_noise = 0
        skipped_domain = 0
        
        for c in chunks:
            text = c.get("text", "")
            # Preserve semantic filtering for pure OCR noise/boilerplate (which are non-retrievable)
            if not is_meaningful_legal_content(text):
                skipped_noise += 1
                continue
            
            # Domain Identification (Flag as non-related instead of skipping)
            if not has_legal_domain_anchors(text):
                c["content_type"] = "non_related"
            else:
                c["content_type"] = "legal"
                
            valid_chunks.append(c)
        
        if skipped_noise > 0:
            print(f"Filtered out {skipped_noise} noise/boilerplate chunks (Signatures, IDs, or Timestamps).")
        
        indexed_legal = sum(1 for c in valid_chunks if c["content_type"] == "legal")
        indexed_non_related = len(valid_chunks) - indexed_legal
        
        if indexed_legal > 0:
            print(f"Indexing {indexed_legal} legal/financial chunks.")
        if indexed_non_related > 0:
            print(f"Indexing {indexed_non_related} non-related chunks (Academic/Admin tagged).")

        if not valid_chunks:
            print(f"No meaningful text found for doc_id {doc_id}. Skipping.")
            return

        # Prepare for bulk embedding (e5-base requires 'passage: ' prefix)
        prefixed_texts = [f"passage: {c['text'].strip()}" for c in valid_chunks]
        
        # Log a preview of the first chunk to ensure we are embedding actual text
        print(f"Sample chunk text for embedding: '{prefixed_texts[0][:60]}...'")
        
        # Ensure we always normalize embeddings for correct Cosine Similarity via Dot Product
        embeddings = self.model.encode(prefixed_texts, normalize_embeddings=True)
        
        db = get_database()
        collection = db[EMBEDDED_COLLECTION_NAME]
        
        batch_records = []
        for i, (chunk, embedding_vec) in enumerate(zip(valid_chunks, embeddings)):
            chunk_text = chunk["text"].strip()
            # Generate deterministic SHA256 ID for the content to prevent cross-doc duplication
            stable_id = hashlib.sha256(chunk_text.encode()).hexdigest()
            
            record = {
                "_id": stable_id,
                "doc_id": doc_id,
                "text": chunk_text if chunk.get("content_type") != "non_related" else None,
                "embedding": embedding_vec.tolist(),
                "metadata": {
                    "source": source,
                    "content_type": chunk.get("content_type", "legal"), # Now explicitly flagged
                    "chunk_id": chunk.get("chunk_id", f"c{i+1}"), # Original sequential ID
                    "timestamp": time.time(),
                    "page": chunk.get("metadata", {}).get("page", 1)
                }
            }
            batch_records.append(record)
            
            # --- Real-time Index Update ---
            self.metadata.append(record)
            new_row = embedding_vec.reshape(1, -1)
            if self.index is None:
                self.index = new_row
            else:
                self.index = np.vstack([self.index, new_row])
        
        if batch_records:
            # Upsert into MongoDB to prevent duplicate SHA256 IDs
            for r in batch_records:
                collection.update_one({"_id": r["_id"]}, {"$set": r}, upsert=True)
                
        print(f"Successfully indexed {len(batch_records)} chunks for doc_id {doc_id}.")

    def query(self, user_query: str, k: int = 3) -> list[dict]:
        """
        Perform a local vector search.
        If local results have low similarity, trigger external augmentation.
        """
        if self.index is None or len(self.index) == 0:
            return []
            
        # e5-base query protocol: must use 'query: ' prefix
        query_vector = self.model.encode([f"query: {user_query}"], normalize_embeddings=True)[0]
        
        # Calculate Cosine Similarities (Dot product since embeddings are normalized)
        # We ensure we look for 'embedding' first
        similarities = np.dot(self.index, query_vector)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            res = self.metadata[idx].copy()
            res["relevance_score"] = score
            results.append(res)
            
        # --- Augmentation Logic (Single Query-time external fetch if needed) ---
        max_score = results[0]["relevance_score"] if results else 0
        if max_score < SIMILARITY_THRESHOLD:
            print(f"Low local confidence ({max_score:.2f}). Triggering external legal augmentation...")
            # (In a real demo, this would call search_legal_context and ingest results)
            
        return results

# Singleton interface for the Orchestrator
rag_manager = LocalRAGManager()

def retrieve_legal_context(query_text: str, k: int = 3):
    """Main interface for LLM/Agentic AI to get background context."""
    return rag_manager.query(query_text, k=k)

if __name__ == "__main__":
    print("Testing In-Memory Vector Search...")
    # Mock data injection
    rag_manager.add_document("test_001", "Regulation 12 covers Margin Trading in NSE for stock brokers.")
    
    # Search
    matches = retrieve_legal_context("What are the rules for margin trading?", k=1)
    if matches:
        print(f"Match Found (Score: {matches[0]['relevance_score']:.2f}): {matches[0]['text']}")
