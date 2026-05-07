import asyncio
import os
import time
import hashlib
import logging
import traceback
from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from .prompts import CLAIM_EXTRACTOR_PROMPT, ENTAILMENT_VERIFIER_PROMPT, CONSOLIDATOR_PROMPT
from Schemas.validation_schema import (
    AnalysisReport, Verdict, Metrics, DocumentMetadata, 
    AdvancedAnalysis, TraceabilityEntry, CharSpan
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hallucination_engine")

load_dotenv()

# --- Constants & DB Config ---
# Re-using the URI from Ingestion/store.py but for Async Motor
MONGO_URI = "mongodb://Tanveer:%40Tanveer12345@ac-rqleiyn-shard-00-00.do3in0d.mongodb.net:27017,ac-rqleiyn-shard-00-01.do3in0d.mongodb.net:27017,ac-rqleiyn-shard-00-02.do3in0d.mongodb.net:27017/?ssl=true&replicaSet=atlas-hszf5p-shard-0&authSource=admin&retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "deep-detector"
STAGED_COLLECTION = "staged_db"
EMBEDDED_COLLECTION = "Embedded_docs_db"
GEMINI_API = os.environ.get("GEMINI_API_KEY") # Centralized API Configuration

# --- Global Rate Limiter State ---
# Strictly below 15 RPM (14 RPM = ~4.28s interval)
LAST_GEMINI_CALL = 0.0
GEMINI_LOCK = asyncio.Lock()

async def enforce_gemini_rate_limit(rpm: float = 14.0):
    """
    Enforces a global rate limit for Gemini API calls to stay strictly below 15 RPM.
    Uses module-level state and an async lock to coordinate between concurrent nodes.
    """
    global LAST_GEMINI_CALL
    interval = 60.0 / rpm
    
    async with GEMINI_LOCK:
        now = time.time()
        elapsed = now - LAST_GEMINI_CALL
        if elapsed < interval:
            delay = interval - elapsed
            logger.info(f"Rate limiting: sleeping for {delay:.2f}s to maintain <15 RPM...")
            await asyncio.sleep(delay)
        
        LAST_GEMINI_CALL = time.time()

# --- Pydantic Models for Internal Logic (If not in validation_schema) ---

class ClaimSpan(BaseModel):
    claim_text: str = Field(description="The exact text quote of the claim extracted from the document.")
    start_char: int = Field(description="The perceived starting character index of the claim.")
    end_char: int = Field(description="The perceived ending character index of the claim.")

class ClaimsList(BaseModel):
    claims: List[ClaimSpan]

# Note: Other models (Verdict, AnalysisReport, etc.) are imported from Schemas.validation_schema

# --- State Definition ---

class GraphState(TypedDict):
    primary_doc_id: str
    primary_doc_text: str
    total_chars: int
    claims: List[ClaimSpan]
    verdicts: List[Verdict]
    final_report: Optional[Dict[str, Any]]

# --- Agent Nodes ---

async def claim_extractor_node(state: GraphState):
    """
    Node 1: Extract testable claims and calculate exact character offsets using a Python safety net.
    """
    print("--- NODE 1: CLAIM EXTRACTOR ---")
    doc_text = state["primary_doc_text"]
    
    # Strictly maintain RPM < 15
    await enforce_gemini_rate_limit()
    
    client = genai.Client(api_key=GEMINI_API)
    
    prompt = f"""
    {CLAIM_EXTRACTOR_PROMPT}
    
    DOCUMENT:
    {doc_text}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClaimsList,
        ),
    )
    
    raw_claims = response.parsed.claims
    refined_claims = []
    
    # --- Python Index Safety Net ---
    for claim in raw_claims:
        # Re-verify the quote exists and lock in exact offsets
        # This prevents LLM "drift" in character counting
        start_idx = doc_text.find(claim.claim_text)
        if start_idx != -1:
            claim.start_char = start_idx
            claim.end_char = start_idx + len(claim.claim_text)
            refined_claims.append(claim)
        else:
            # Fallback if the LLM hallucinated the quote slightly
            print(f"Warning: Could not anchor claim: {claim.claim_text[:30]}...")
            refined_claims.append(claim)
            
    return {
        "claims": refined_claims,
        "total_chars": len(doc_text)
    }

async def vector_retrieval_node(state: GraphState):
    """
    Node 2: Retrieve evidence from Vector DB with strict security filters using Motor (Async).
    """
    print("--- NODE 2: ASYNC VECTOR RETRIEVAL ---")
    doc_id = state["primary_doc_id"]
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DATABASE_NAME]
    
    # 🛡️ Security Filter: Must be 'supported' and MUST NOT be the current doc_id
    security_filter = {
        "metadata.supported": True,
        "doc_id": {"$ne": doc_id}
    }
    
    # In a real system, we'd loop through each claim's embedding here
    all_verdicts = []
    
    for claim in state["claims"]:
        # Mocking the Vector Search -> Metadata Resolve flow
        cursor = db[STAGED_COLLECTION].find(security_filter).limit(3)
        chunks = await cursor.to_list(length=3)
        evidence = [
            TraceabilityEntry(
                chunk_id=c.get("chunk_id", "ERR"),
                evidence_text=c.get("text", ""),
                cloudinary_url=c.get("cloudinary_url"),
                page_number=c.get("page"),
                section=c.get("section")
            ) for c in chunks
        ]
        
        # --- Call Verification Logic for this claim ---
        # (This is the logic previously in Node 3, now integrated sequentially for reliability)
        evidence_text = "\n\n".join([f"Source [{e.chunk_id}]: {e.evidence_text}" for e in evidence])
        
        # Strictly maintain RPM < 15
        await enforce_gemini_rate_limit()
        
        v_client = genai.Client(api_key=GEMINI_API)
        v_prompt = f"""
        {ENTAILMENT_VERIFIER_PROMPT}
        
        CLAIM:
        {claim.claim_text}
        
        EVIDENCE:
        {evidence_text}
        """
        
        v_response = v_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=v_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Verdict,
            ),
        )
        
        verdict = v_response.parsed
        verdict.claim_id = f"clm_{hashlib.md5(claim.claim_text.encode()).hexdigest()[:8]}"
        verdict.char_span = CharSpan(start=claim.start_char, end=claim.end_char)
        verdict.traceability_matrix = evidence # Attach the evidence chunks
        
        all_verdicts.append(verdict)
        
    return {"verdicts": all_verdicts}


async def web_escalation_node(state: GraphState):
    """
    Node 4: Mocked web escalation strictly limited to .gov.in domains.
    """
    print("--- NODE 4: WEB ESCALATION ---")
    # This node is triggered only if a claim is NEUTRAL.
    # In a real system, this would call an async search tool.
    pass

async def final_judge_node(state: GraphState):
    """
    Node 5: Aggregator & Metrics Engine. Calculates percentages based on character spans.
    Utilizes Gemini's native response_schema for the final AnalysisReport.
    """
    print("--- NODE 5: FINAL JUDGE & AGGREGATOR ---")
    total_chars = state["total_chars"]
    verdicts = state["verdicts"]
    
    hallucinated_chars = 0
    unverified_chars = 0
    
    for v in verdicts:
        # Calculate length based on strictly the char_span
        span_len = v.char_span.end - v.char_span.start
        if v.status == "CONTRADICTED":
            hallucinated_chars += span_len
        elif v.status == "UNVERIFIED":
            unverified_chars += span_len
            
    hallucination_rate = (hallucinated_chars / total_chars) * 100 if total_chars > 0 else 0
    unverified_rate = (unverified_chars / total_chars) * 100 if total_chars > 0 else 0
    
    # --- Native Gemini Consolidation (Auditor Persona) ---
    client = genai.Client(api_key=GEMINI_API)
    
    # Prepare the context for the consolidator
    verdicts_summary = []
    for v in verdicts:
        verdicts_summary.append({
            "claim": v.claim_text,
            "status": v.status,
            "is_flagged": v.is_flagged,
            "flag_reason": v.flag_reason,
            "reasoning": v.advanced_analysis.agent_reasoning
        })
    
    prompt = f"""
    {CONSOLIDATOR_PROMPT}
    
    ANALYSIS RESULTS TO CONSOLIDATE:
    {verdicts_summary}
    """
    
    # Strictly maintain RPM < 15
    await enforce_gemini_rate_limit()
    
    # API call with strict response_schema (Structured Outputs)
    # The SDK handles the JSON parsing and schema validation internally.
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisReport,
            ),
        )
        
        final_report = response.parsed
        
        # Override metadata and metrics with calculated data to ensure absolute accuracy
        final_report.document_metadata = DocumentMetadata(
            primary_doc_id=state["primary_doc_id"],
            uploaded_by="SYSTEM_AGENT", # Or fetch from elsewhere
            analysis_timestamp=time.ctime(),
            status="COMPLETED"
        )
        final_report.metrics = Metrics(
            total_chars=total_chars,
            hallucinated_chars=hallucinated_chars,
            unverified_chars=unverified_chars,
            hallucination_rate_percentage=round(hallucination_rate, 2),
            unverified_rate_percentage=round(unverified_rate, 2)
        )
        
        # Re-attach the detailed verdicts
        final_report.verdicts = verdicts
        
        return {"final_report": final_report.model_dump()}
        
    except Exception as e:
        logger.error(f"Failed to consolidate report: {str(e)}")
        # Fallback to minimal report if API fails
        return {
            "final_report": {
                "document_metadata": {"primary_doc_id": state["primary_doc_id"], "status": "FAILED"},
                "executive_summary": "Failed to generate consolidated summary due to API error.",
                "metrics": {},
                "verdicts": []
            }
        }

# --- LangGraph Construction ---

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("claim_extractor", claim_extractor_node)
    workflow.add_node("vector_retrieval", vector_retrieval_node)
    workflow.add_node("final_judge", final_judge_node)
    
    # Map-Reduce Step (Node 3) is usually implemented via Send or parallel edges
    # For this script, we'll use a direct sequential call to demonstrate logic,
    # though LangGraph's MAP system is intended for production parallelization.
    
    workflow.set_entry_point("claim_extractor")
    workflow.add_edge("claim_extractor", "vector_retrieval")
    workflow.add_edge("vector_retrieval", "final_judge")
    
    return workflow.compile()

async def analyze_document_hallucinations(doc_id: str, text: str):
    """Main execution entry point."""
    try:
        logger.info(f"Starting hallucination analysis for doc_id: {doc_id}")
        app = build_graph()
        initial_state = {
            "primary_doc_id": doc_id,
            "primary_doc_text": text,
            "verdicts": [],
            "claims": [],
            "total_chars": 0,
            "final_report": None
        }
        
        final_output = await app.ainvoke(initial_state)
        
        # Validate the final structure against Pydantic before returning
        report = final_output.get("final_report")
        if not report:
            raise ValueError("Graph execution finished but generated no final_report.")
            
        return report
        
    except Exception as e:
        # Local logging with full stack trace for DevOps/Debugging
        logger.error(f"Critical failure during hallucination analysis for doc_id {doc_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Construct the secure, concise FAILED payload for the main backend
        error_code = "INTERNAL_ENGINE_ERROR"
        if "rate limit" in str(e).lower(): error_code = "LLM_RATE_LIMIT"
        elif "timeout" in str(e).lower(): error_code = "LLM_TIMEOUT"
        elif "validation" in str(e).lower(): error_code = "SCHEMA_VALIDATION_ERROR"
        
        return {
            "document_metadata": {
                "primary_doc_id": doc_id,
                "analysis_timestamp": time.ctime(),
                "status": "FAILED"
            },
            "error_code": error_code,
            "error_message": f"Hallucination analysis failed: {str(e)}"
        }


